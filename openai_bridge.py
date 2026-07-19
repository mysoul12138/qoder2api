"""
OpenAI 兼容 API 桥接服务器(入口 + 业务编排)。

业务逻辑分层:
  - models.py      模型映射
  - transform.py   OpenAI↔Qoder 消息转换 + 流式累积器
  - qoder_auth.py  Qoder 认证/加密/HTTP 客户端

本文件保留: OpenAiBridge(session 管理 + 流式/同步转发)、
            BridgeRegistry(多 PAT 实例池)、Flask 路由与 main。
"""

import base64
import copy
import hashlib
import json
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable
from flask import Flask, Response, jsonify, request, stream_with_context
import qoder_auth
from qoder_auth import AuthIdentity
import models
from transform import (
    BridgeDelta,
    StreamAccumulator,
    ToolCallAccumulator,
    _apply_openai_tool_config,
    _build_qoder_messages,
    _extract_delta,
    _extract_latest_user_prompt,
    _extract_message_images,
    _make_chunk,
    parse_tool_calls_text,
)


_REFRESH_MARGIN_MS = 2 * 3600 * 1000  # 2 hours
_CATALOG_TTL = 600  # 模型目录动态拉取缓存秒数


class OpenAiBridge:
    def __init__(self, pat: str, region: qoder_auth.RegionConfig | None = None):
        self._lock = (
            threading.Lock()
        )  # guards sess/identity/tokens/expiry/generation/template
        self._refresh_lock = threading.Lock()  # serializes renewal network calls
        self.region = region or qoder_auth.CN
        self._pat = pat

        # Stable machine identity for the lifetime of this bridge. Only the
        # session token (jt-/jrt-) rotates on renewal, not the machine id.
        self.machine_id = str(uuid.uuid4())
        raw_token = (uuid.uuid4().hex + uuid.uuid4().hex)[:50]
        self.machine_token = (
            base64.urlsafe_b64encode(raw_token.encode()).decode().rstrip("=")
        )
        self.machine_type = uuid.uuid4().hex[:18]

        # Session state (guarded by self._lock)
        self.sess: qoder_auth.SessionContext | None = None
        self.identity: AuthIdentity | None = None
        self.template_base: dict | None = None
        self._refresh_token: str = ""
        self._security_oauth_token: str = ""
        self._expire_time_ms: int = 0

        # 模型目录动态缓存 (guarded by self._catalog_lock)
        self._catalog: models.ModelCatalog | None = None
        self._catalog_ts: float = 0.0
        self._catalog_lock = threading.Lock()

        self._bootstrap_session()

    # ─────────────────────────────────────────────
    # Session lifecycle (cold exchange + renewal)
    # ─────────────────────────────────────────────

    def _bootstrap_session(self) -> None:
        """Cold PAT → jobToken exchange on first use."""
        jt = qoder_auth.exchange_job_token(
            self._pat,
            self.machine_id,
            self.machine_token,
            self.machine_type,
            region=self.region,
        )
        print(
            f"[bridge] session for {jt.get('name', '')} ({jt.get('id', '')}) "
            f"[{self.region.name}] exp={jt.get('expireTime')}"
        )
        self._apply_job_token(jt)
        self._load_template()

    def _apply_job_token(self, jt: dict) -> None:
        """Build sess/identity from a jobToken response and store latest tokens."""
        identity = AuthIdentity(
            name=jt.get("name", ""),
            aid=jt.get("id", ""),
            uid=jt.get("id", ""),
            yx_uid="",
            organization_id="",
            organization_name="",
            user_type=jt.get("userType", "personal_standard"),
            security_oauth_token=jt.get("securityOauthToken", ""),
            refresh_token=jt.get("refreshToken", ""),
        )
        sess = qoder_auth.new_session(
            identity, self.machine_id, self.machine_token, self.machine_type
        )
        with self._lock:
            self.sess = sess
            self.identity = identity
            self._refresh_token = identity.refresh_token
            self._security_oauth_token = identity.security_oauth_token
            self._expire_time_ms = int(jt.get("expireTime") or 0)

    def _load_template(self) -> None:
        # 加载 baseprompt.json 模板（每个 Bridge 实例独立副本）。
        # 用绝对路径，避免 CWD != 项目根时 FileNotFoundError（如 systemd / 容器
        # 用 WorkingDirectory 启动、或 `python -m openai_bridge` 场景）。
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "baseprompt.json"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            base_prompt = f.read()
        # 模板里的 {UUID1..5} / {TIME1} 占位符 *不需要* 在这里替换：handle_chat
        # 会用 runtime 值覆盖 request_id / session_id / chat_record_id /
        # request_set_id / business.id / business.begin_at 这全部 6 个字段。
        # 早期版本在这里做了一次替换，纯属白做（且每个实例只跑一次，意味着同一实例
        # 的所有请求共享同一组随机 UUID —— 反而是 bug 隐患）。已移除。
        with self._lock:
            self.template_base = json.loads(base_prompt)

    def _needs_refresh(self) -> bool:
        now_ms = int(time.time() * 1000)
        with self._lock:
            return (
                self.sess is None
                or self._expire_time_ms == 0
                or now_ms > self._expire_time_ms - _REFRESH_MARGIN_MS
            )

    def _do_renew(self, *, force: bool) -> None:
        """Renew the short-lived session token via refreshToken (scheme B).

        Falls back to a cold PAT exchange if the refresh token is rejected.
        Caller must NOT hold ``self._lock``.
        """
        with self._lock:
            rt = self._refresh_token
            sot = self._security_oauth_token
        try:
            jt = qoder_auth.refresh_job_token(
                self._pat,
                rt,
                sot,
                self.machine_id,
                self.machine_token,
                self.machine_type,
                region=self.region,
            )
            label = "refreshed" if not force else "force-refreshed"
            print(f"[bridge] session {label} (exp={jt.get('expireTime')})")
        except qoder_auth.QoderAuthError as e:
            if not force:
                # proactive refresh rejected; surface so caller can react
                raise
            print(f"[bridge] refresh rejected ({e}); falling back to PAT exchange")
            jt = qoder_auth.exchange_job_token(
                self._pat,
                self.machine_id,
                self.machine_token,
                self.machine_type,
                region=self.region,
            )
            print(f"[bridge] PAT re-exchange ok (exp={jt.get('expireTime')})")
        self._apply_job_token(jt)

    def ensure_fresh_session(self) -> None:
        """Proactively rotate the session token before its 24h expiry."""
        if not self._needs_refresh():
            return
        with self._refresh_lock:
            # re-check after acquiring: another request may have just renewed
            if not self._needs_refresh():
                return
            self._do_renew(force=False)

    def _force_refresh(self) -> None:
        """Reactive renewal after a 401 (ignore remaining TTL)."""
        with self._refresh_lock:
            self._do_renew(force=True)

    def _current_sess(self) -> qoder_auth.SessionContext:
        with self._lock:
            sess = self.sess
        assert sess is not None, "session not bootstrapped"
        return sess

    def get_catalog(self) -> models.ModelCatalog:
        """动态模型目录 (TTL 缓存)。优先从 /api/v2/model/list 拉取; 失败回退内置表。

        回退是显式的 (打印 WARN), 不会静默 —— 方便从日志确认动态加载是否生效。
        """
        now = time.time()
        with self._catalog_lock:
            cached = self._catalog
            if cached is not None and now - self._catalog_ts < _CATALOG_TTL:
                return cached
        fetched = None
        try:
            self.ensure_fresh_session()
            raw = qoder_auth.fetch_model_catalog(self._current_sess(), self.region)
            fetched = models.extract_catalog(raw) if isinstance(raw, dict) else None
            if fetched:
                print(
                    f"[models] dynamic catalog loaded: {len(fetched.keys())} models "
                    f"[{', '.join(fetched.keys())}]"
                )
            else:
                print(
                    "[models] WARN model/list response shape unexpected; using fallback"
                )
        except Exception as e:
            print(f"[models] WARN dynamic fetch failed ({e!r}); using fallback")
        cat = fetched or models.default_catalog()
        with self._catalog_lock:
            self._catalog = cat
            self._catalog_ts = now
        return cat

    def _open_stream_with_retry(
        self,
        url: str,
        body: dict,
        extra_headers: dict | None,
        on_line: Callable[[str], None],
    ) -> None:
        """open_stream_lines with one auth-refresh retry on 401/403."""
        for attempt in (1, 2):
            sess = self._current_sess()
            try:
                qoder_auth.open_stream_lines(sess, url, body, extra_headers, on_line)
                return
            except qoder_auth.QoderAuthError as e:
                if attempt >= 2:
                    raise
                print(
                    f"[bridge] auth error mid-stream ({e}); refreshing and retrying once"
                )
                try:
                    self._force_refresh()
                except Exception as rf:
                    print(f"[bridge] reactive refresh failed: {rf}")
                    raise

    def handle_chat(self, req_body: dict) -> Response:
        # Proactively rotate the 24h session token before it expires so the
        # same PAT keeps working across day boundaries (scheme B).
        self.ensure_fresh_session()
        stream = req_body.get("stream", False)
        model_param = req_body.get("model")
        catalog = self.get_catalog()
        openai_model, qoder_model = models.resolve_model(model_param, catalog)
        messages = req_body.get("messages", [])
        body = copy.deepcopy(self.template_base)
        nid = str(uuid.uuid4())
        body["request_id"] = nid
        body["chat_record_id"] = nid
        body["request_set_id"] = str(uuid.uuid4())
        body["session_id"] = str(uuid.uuid4())
        body["stream"] = True
        body["aliyun_user_type"] = self.identity.user_type
        body["model_config"]["key"] = qoder_model
        body["model_config"]["is_reasoning"] = True
        body["chat_context"]["extra"]["modelConfig"]["key"] = qoder_model
        body["chat_context"]["extra"]["modelConfig"]["is_reasoning"] = True
        body["business"]["id"] = str(uuid.uuid4())
        body["business"]["begin_at"] = int(time.time() * 1000)

        prompt = _extract_latest_user_prompt(messages)
        body["chat_context"]["text"]["text"] = prompt
        body["chat_context"]["extra"]["originalContent"]["text"] = prompt
        body["business"]["name"] = prompt[:30] if len(prompt) > 30 else prompt

        tools_enabled = _apply_openai_tool_config(body, req_body)
        body["messages"] = _build_qoder_messages(messages, prompt, tools_enabled)

        # multimodal: if any user message carries images, gate on vision-
        # capable models. Only Qwen3.7-Plus supports image input; Max does not.
        has_images = any(
            _extract_message_images(m) for m in messages if isinstance(m, dict)
        )
        if has_images:
            if openai_model not in catalog.vision_models:
                supported = ", ".join(sorted(catalog.vision_models)) or "(none)"
                raise ValueError(
                    f"Image input is not supported by model '{openai_model}'. "
                    f"Use one of: {supported}."
                )
            body["model_config"]["is_vl"] = True
            body["chat_context"]["extra"]["modelConfig"]["is_vl"] = True
            img_count = sum(
                len(_extract_message_images(m)) for m in messages if isinstance(m, dict)
            )
            print(
                f"[bridge] multimodal: {img_count} image(s) attached [{openai_model}]"
            )

        # NOTE: 故意不打印 prompt 内容,避免用户输入(可能含 PII/密钥)落入容器日志。
        # 仅记录长度和模型名,足以用于排障且不泄漏内容。
        print(f"[bridge] chat req: prompt_len={len(prompt)} model={openai_model}")

        url = qoder_auth.chat_url(self.region)
        extra_headers = {
            "x-model-key": qoder_model,
            "x-model-source": body["model_config"].get("source", "system"),
        }

        req_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if stream:
            return self._handle_stream(
                body, url, extra_headers, req_id, created, openai_model, tools_enabled
            )
        return self._handle_sync(
            body, url, extra_headers, req_id, created, openai_model, tools_enabled
        )

    def _handle_stream(
        self, body, url, extra_headers, req_id, created, model, tools_enabled
    ) -> Response:
        chunk_queue: queue.Queue[str | None] = queue.Queue()
        acc = StreamAccumulator(
            req_id, created, model, tools_enabled, emit_fn=lambda c: chunk_queue.put(c)
        )

        def reader():
            try:

                def on_line(line: str):
                    if not line.startswith("data:"):
                        return
                    delta = _extract_delta(line[5:].strip())
                    if not delta.is_empty():
                        acc.accept(delta)

                self._open_stream_with_retry(url, body, extra_headers, on_line)
                acc.flush()

                # finish chunk
                done = _make_chunk(req_id, created, model)
                done["choices"][0]["finish_reason"] = acc.finish_reason()
                done["choices"][0]["delta"] = {}
                chunk_queue.put(f"data: {json.dumps(done, ensure_ascii=False)}\n\n")
            except Exception as e:
                # 必须把错误暴露给客户端,否则 OpenAI 兼容客户端会把截断的流当成正常结束。
                # 在 EOF sentinel 前先注入一个携带 error 字段且 finish_reason="error"
                # 的 chunk,让上游能感知失败并向用户报告。
                print(f"[bridge] stream error: {e}")
                try:
                    err_chunk = _make_chunk(req_id, created, model)
                    err_chunk["choices"][0]["finish_reason"] = "error"
                    err_chunk["choices"][0]["delta"] = {}
                    err_chunk["error"] = {
                        "message": str(e),
                        "type": "qoder_error",
                    }
                    chunk_queue.put(
                        f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                    )
                except Exception as inner:
                    # 兜底:即便错误序列化失败也保证不会卡住 generator。
                    print(f"[bridge] failed to emit stream error chunk: {inner}")
            finally:
                chunk_queue.put(None)  # EOF sentinel

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        def generate():
            while True:
                try:
                    chunk = chunk_queue.get(timeout=300)
                except queue.Empty:
                    break
                if chunk is None:
                    break
                yield chunk
            yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    def _handle_sync(
        self, body, url, extra_headers, req_id, created, model, tools_enabled
    ) -> Response:
        full_content: list[str] = []
        full_reasoning_content: list[str] = []
        tool_calls = ToolCallAccumulator()

        def on_line(line: str):
            if not line.startswith("data:"):
                return
            delta = _extract_delta(line[5:].strip())
            if delta.reasoning_content:
                full_reasoning_content.append(delta.reasoning_content)
            if delta.content:
                full_content.append(delta.content)
            if delta.tool_calls and len(delta.tool_calls) > 0:
                tool_calls.append(delta.tool_calls)

        self._open_stream_with_retry(url, body, extra_headers, on_line)

        full_text = "".join(full_content)
        fallback_tool_calls = None
        if tool_calls.is_empty() and tools_enabled:
            fallback_tool_calls = parse_tool_calls_text(full_text)

        msg: dict = {"role": "assistant"}
        if fallback_tool_calls is not None:
            msg["content"] = None
            msg["tool_calls"] = fallback_tool_calls
        elif not full_text and not tool_calls.is_empty():
            msg["content"] = None
        else:
            msg["content"] = full_text
        if full_reasoning_content:
            msg["reasoning_content"] = "".join(full_reasoning_content)

        if not tool_calls.is_empty():
            msg["tool_calls"] = tool_calls.snapshot()

        finish_reason = (
            "tool_calls"
            if (not tool_calls.is_empty() or fallback_tool_calls is not None)
            else "stop"
        )

        out = {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": msg,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        return jsonify(out)


class BridgeRegistry:
    """线程安全的多区域 Bridge 实例注册表。"""

    def __init__(self) -> None:
        self._bridges: dict[str, OpenAiBridge] = {}
        self._lock = threading.Lock()

    def _key(self, region_name: str, pat: str) -> str:
        raw = f"{region_name}:{pat}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get_or_create(self, pat: str, region: qoder_auth.RegionConfig) -> OpenAiBridge:
        key = self._key(region.name, pat)
        bridge = self._bridges.get(key)
        if bridge is not None:
            return bridge
        with self._lock:
            # double-check after acquiring lock
            bridge = self._bridges.get(key)
            if bridge is not None:
                return bridge
            bridge = OpenAiBridge(pat, region=region)
            self._bridges[key] = bridge
            return bridge


_registry: BridgeRegistry | None = None


def _get_setting(key: str) -> str | None:
    return os.environ.get(key)


def _resolve_port() -> int:
    port = _get_setting("QODER_PORT")
    if not port:
        return 8963
    return int(port)


def _extract_pat_from_request() -> str | None:
    """从 Authorization: Bearer <PAT> 提取 PAT。"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        if token:
            return token
    return None


def create_app() -> Flask:
    global _registry
    _registry = BridgeRegistry()

    app = Flask(__name__)

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        try:
            raw_pat = _extract_pat_from_request()
            if not raw_pat:
                return jsonify(
                    {
                        "error": {
                            "message": "Missing Authorization: Bearer <PAT>",
                            "type": "invalid_request_error",
                        }
                    }
                ), 401
            real_pat, region = qoder_auth.resolve(raw_pat)
            bridge = _registry.get_or_create(real_pat, region)
            req_body = request.get_json(force=True)
            return bridge.handle_chat(req_body)
        except ValueError as e:
            return jsonify(
                {"error": {"message": str(e), "type": "invalid_request_error"}}
            ), 400
        except Exception as e:
            return jsonify({"error": {"message": str(e), "type": "qoder_error"}}), 500

    # 注意:函数名不能叫 `models`,否则会遮蔽顶部 `import models`,
    # 任何后续在本文件内使用 `models.xxx` 都会变成 AttributeError。
    @app.route("/v1/models", methods=["GET"])
    def list_models():
        # 带 PAT 则返回该账户动态拉取的目录; 无 PAT 回退内置表。
        raw_pat = _extract_pat_from_request()
        if raw_pat:
            try:
                real_pat, region = qoder_auth.resolve(raw_pat)
                bridge = _registry.get_or_create(real_pat, region)
                return jsonify(models.models_payload(bridge.get_catalog()))
            except Exception as e:
                print(f"[models] WARN /v1/models dynamic failed ({e!r}); fallback")
        return jsonify(models.models_payload())

    return app


def main():
    host = _get_setting("QODER_HOST") or "127.0.0.1"
    port = _resolve_port()
    app = create_app()
    print(f"[bridge] listening http://{host}:{port}/v1/chat/completions")
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()

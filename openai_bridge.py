"""
OpenAI 兼容 API 桥接服务器(入口 + 业务编排)。

业务逻辑分层:
  - models.py      模型映射
  - transform.py   OpenAI↔Qoder 消息转换 + 流式累积器
  - qoder_auth.py  Qoder 认证/加密/HTTP 客户端

本文件保留: OpenAiBridge(session 管理 + 流式/同步转发)、
            BridgeRegistry(多 PAT 实例池)、FastAPI 路由与 main。
"""

import asyncio
import base64
import contextlib
import copy
import hashlib
import json
import os
import threading
import time
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import checkin
import qoder_auth
from qoder_auth import AuthIdentity
import models
from transform import (
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

# Bridge 注册表驱逐策略 (防止公开部署时 PAT 数无界增长导致内存堆积)
_BRIDGE_TTL_SEC = 30 * 60  # 空闲超过 30 分钟 → 下次 sweep 淘汰
_BRIDGE_MAX_ENTRIES = 1024  # 硬上限; 超出按最久未访问淘汰 (无视 TTL)
_BRIDGE_SWEEP_EVERY = 64  # 每 N 次新建触发一次惰性 sweep


async def _sse_drain(aq: "asyncio.Queue[str | None]", task: "asyncio.Task[None]"):
    """把 producer 写入队列的 SSE chunk 逐个 yield 给 StreamingResponse。

    收到 EOF sentinel 后发 [DONE]; finally 确保 producer task 被回收
    (客户端中途断连时取消, 避免泄漏)。
    """
    try:
        while True:
            chunk = await aq.get()
            if chunk is None:
                break
            yield chunk
        yield "data: [DONE]\n\n"
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(Exception):
            await task


class OpenAiBridge:
    def __init__(self, pat: str, region: qoder_auth.RegionConfig | None = None):
        self._lock = (
            threading.Lock()
        )  # guards sess/identity/tokens/expiry/generation/template
        self._refresh_lock = asyncio.Lock()  # serializes renewal network calls
        # 注意: 必须用 asyncio.Lock 而非 threading.Lock —— 续期锁内需 await
        # 网络调用, threading.Lock 被持有时其他协程 acquire 会阻塞整个事件循环。
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
        self._bootstrapped = False

    # ─────────────────────────────────────────────
    # Session lifecycle (cold exchange + renewal)
    # ─────────────────────────────────────────────

    async def _bootstrap_session(self) -> None:
        """Cold PAT → jobToken exchange (lazy, first use)."""
        jt = await qoder_auth.exchange_job_token(
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
        self._bootstrapped = True

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
            raw_exp = jt.get("expireTime") or 0
            try:
                self._expire_time_ms = int(raw_exp)
            except (TypeError, ValueError):
                self._expire_time_ms = 0

    def _load_template(self) -> None:
        # 加载 baseprompt.json 模板（每个 Bridge 实例独立副本）。
        # 用绝对路径，避免 CWD != 项目根时 FileNotFoundError（如 systemd / 容器
        # 用 WorkingDirectory 启动、或 `python -m openai_bridge` 场景）。
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "baseprompt.json"
        )
        try:
            with open(template_path, encoding="utf-8") as f:
                base_prompt = f.read()
            template = json.loads(base_prompt)
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"failed to load {template_path}: {e}") from e
        # 模板里的 {UUID1..5} / {TIME1} 占位符 *不需要* 在这里替换：handle_chat
        # 会用 runtime 值覆盖 request_id / session_id / chat_record_id /
        # request_set_id / business.id / business.begin_at 这全部 6 个字段。
        # 早期版本在这里做了一次替换，纯属白做（且每个实例只跑一次，意味着同一实例
        # 的所有请求共享同一组随机 UUID —— 反而是 bug 隐患）。已移除。
        with self._lock:
            self.template_base = template

    def _needs_refresh(self) -> bool:
        now_ms = time.time_ns() // 1_000_000
        with self._lock:
            return (
                self.sess is None
                or self._expire_time_ms == 0
                or now_ms > self._expire_time_ms - _REFRESH_MARGIN_MS
            )

    async def _do_renew(self, *, force: bool) -> None:
        """Renew the short-lived session token via refreshToken (scheme B).

        Falls back to a cold PAT exchange if the refresh token is rejected.
        Caller must NOT hold ``self._lock``.
        """
        with self._lock:
            rt = self._refresh_token
            sot = self._security_oauth_token
        try:
            jt = await qoder_auth.refresh_job_token(
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
            jt = await qoder_auth.exchange_job_token(
                self._pat,
                self.machine_id,
                self.machine_token,
                self.machine_type,
                region=self.region,
            )
            print(f"[bridge] PAT re-exchange ok (exp={jt.get('expireTime')})")
        self._apply_job_token(jt)

    async def ensure_fresh_session(self) -> None:
        """Proactively rotate the session token before its 24h expiry."""
        if self._bootstrapped and not self._needs_refresh():
            return
        async with self._refresh_lock:
            # re-check after acquiring: another request may have just renewed
            if not self._bootstrapped:
                await self._bootstrap_session()
                return
            if not self._needs_refresh():
                return
            await self._do_renew(force=False)

    async def _force_refresh(self) -> None:
        """Reactive renewal after a 401 (ignore remaining TTL)."""
        async with self._refresh_lock:
            await self._do_renew(force=True)

    def _current_sess(self) -> qoder_auth.SessionContext:
        with self._lock:
            sess = self.sess
        assert sess is not None, "session not bootstrapped"
        return sess

    async def get_catalog(self) -> models.ModelCatalog:
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
            await self.ensure_fresh_session()
            raw = await qoder_auth.fetch_model_catalog(
                self._current_sess(), self.region
            )
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

    async def _open_stream_async(
        self,
        url: str,
        body: dict,
        extra_headers: dict | None,
    ):
        """open_stream_lines + 一次鉴权刷新重试 (异步生成器)。

        仅在**尚未产出任何内容行**时重试。一旦已 yield 过真实数据,
        重试会让上游重发整个 prompt 重新生成, 客户端将收到重复内容 ——
        此时直接抛错, 由 _handle_stream 的 producer 走错误分支
        (发一个 finish_reason='error' 的 chunk 再 [DONE])。
        """
        for attempt in (1, 2):
            sess = self._current_sess()
            produced = False
            try:
                async for line in qoder_auth.open_stream_lines(
                    sess, url, body, extra_headers
                ):
                    produced = True
                    yield line
                return
            except qoder_auth.QoderAuthError as e:
                # 两个条件分开判断 (避免 except 体内出现 boolean_operator):
                # - produced: 已产出内容, 重试会重复 → 直接抛错走错误分支
                # - attempt >= 2: 重试次数用尽
                if produced:
                    raise
                if attempt >= 2:
                    raise
                print(
                    f"[bridge] auth error before any content ({e}); "
                    "refreshing and retrying once"
                )
                try:
                    await self._force_refresh()
                except Exception as rf:
                    print(f"[bridge] reactive refresh failed: {rf}")
                    raise

    async def handle_chat(self, req_body: dict):
        # Proactively rotate the 24h session token before it expires so the
        # same PAT keeps working across day boundaries (scheme B).
        await self.ensure_fresh_session()
        # ensure_fresh_session 返回后会话必然已 bootstrap; 显式检查以满足类型收窄。
        if self.template_base is None or self.identity is None:
            raise RuntimeError("session not bootstrapped")
        stream = req_body.get("stream", False)
        model_param = req_body.get("model")
        catalog = await self.get_catalog()
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
        body["business"]["begin_at"] = time.time_ns() // 1_000_000

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
        created = time.time_ns() // 1_000_000_000

        if stream:
            return await self._handle_stream(
                body, url, extra_headers, req_id, created, openai_model, tools_enabled
            )
        return await self._handle_sync(
            body, url, extra_headers, req_id, created, openai_model, tools_enabled
        )

    async def _handle_stream(
        self, body, url, extra_headers, req_id, created, model, tools_enabled
    ) -> StreamingResponse:
        aq: asyncio.Queue[str | None] = asyncio.Queue()
        acc = StreamAccumulator(
            req_id, created, model, tools_enabled, emit_fn=lambda c: aq.put_nowait(c)
        )

        async def producer():
            try:
                async for line in self._open_stream_async(url, body, extra_headers):
                    if not line.startswith("data:"):
                        continue
                    delta = _extract_delta(line[5:].strip())
                    if not delta.is_empty():
                        acc.accept(delta)
                acc.flush()

                # finish chunk
                done = _make_chunk(req_id, created, model)
                done["choices"][0]["finish_reason"] = acc.finish_reason()
                done["choices"][0]["delta"] = {}
                await aq.put(f"data: {json.dumps(done, ensure_ascii=False)}\n\n")
            except Exception as e:
                # 必须把错误暴露给客户端,否则 OpenAI 兼容客户端会把截断的流当成正常结束。
                print(f"[bridge] stream error: {e}")
                try:
                    err_chunk = _make_chunk(req_id, created, model)
                    err_chunk["choices"][0]["finish_reason"] = "error"
                    err_chunk["choices"][0]["delta"] = {}
                    err_chunk["error"] = {"message": str(e), "type": "qoder_error"}
                    await aq.put(
                        f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                    )
                except Exception as inner:
                    # 兜底:即便错误序列化失败也保证不会卡住 generator。
                    print(f"[bridge] failed to emit stream error chunk: {inner}")
            finally:
                await aq.put(None)  # EOF sentinel

        task = asyncio.create_task(producer())

        return StreamingResponse(
            _sse_drain(aq, task),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    async def _handle_sync(
        self, body, url, extra_headers, req_id, created, model, tools_enabled
    ) -> dict:
        full_content: list[str] = []
        full_reasoning_content: list[str] = []
        tool_calls = ToolCallAccumulator()

        async for line in self._open_stream_async(url, body, extra_headers):
            if not line.startswith("data:"):
                continue
            delta = _extract_delta(line[5:].strip())
            if delta.reasoning_content:
                full_reasoning_content.append(delta.reasoning_content)
            if delta.content:
                full_content.append(delta.content)
            if delta.tool_calls and len(delta.tool_calls) > 0:
                tool_calls.append(delta.tool_calls)

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
        return out


class _BridgeEntry:
    """注册表条目: Bridge + 最近访问时间。"""

    __slots__ = ("bridge", "last_access")

    def __init__(self, bridge: "OpenAiBridge") -> None:
        self.bridge = bridge
        self.last_access = time.time()


class BridgeRegistry:
    """线程安全的多区域 Bridge 实例注册表 (TTL + 容量上限驱逐)。

    驱逐策略:
    - TTL: 空闲超过 _BRIDGE_TTL_SEC 的条目, 在下次惰性 sweep 时删除。
    - 容量: 条目数超过 _BRIDGE_MAX_ENTRIES 时, 按最久未访问淘汰, 无视 TTL。

    安全性: 正在处理请求的 Bridge 即便被淘汰也不会被中断 —— 调用方仍持有
    引用, 由 GC 在请求结束后回收; 只是该 PAT 的下一次请求会新建 Bridge。
    极端情况下 (同一 PAT 淘汰瞬间又有新请求) 可能触发重复冷交换, 上游或
    失效旧会话, 表现为单次请求 401, 可自愈。
    """

    def __init__(self) -> None:
        self._bridges: dict[str, _BridgeEntry] = {}
        self._lock = threading.Lock()
        self._sweep_counter = 0

    def _key(self, region_name: str, pat: str) -> str:
        raw = f"{region_name}:{pat}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get_or_create(self, pat: str, region: qoder_auth.RegionConfig) -> OpenAiBridge:
        key = self._key(region.name, pat)
        now = time.time()
        with self._lock:
            entry = self._bridges.get(key)
            if entry is None:
                # 硬上限: 插入前若已满, 先淘汰最久未访问的一个
                if len(self._bridges) >= _BRIDGE_MAX_ENTRIES:
                    self._evict_oldest_locked()
                entry = _BridgeEntry(OpenAiBridge(pat, region=region))
                self._bridges[key] = entry
                self._sweep_counter += 1
                if self._sweep_counter >= _BRIDGE_SWEEP_EVERY:
                    self._sweep_counter = 0
                    self._sweep_idle_locked(now)
            entry.last_access = now
            return entry.bridge

    def _evict_oldest_locked(self) -> None:
        """淘汰最近最少访问的一个条目 (调用方已持锁)。"""
        if not self._bridges:
            return
        oldest = min(self._bridges, key=lambda k: self._bridges[k].last_access)
        del self._bridges[oldest]

    def _sweep_idle_locked(self, now: float) -> None:
        """TTL 淘汰: 删除空闲超过 _BRIDGE_TTL_SEC 的条目 (调用方已持锁)。"""
        stale = [
            k for k, e in self._bridges.items() if now - e.last_access > _BRIDGE_TTL_SEC
        ]
        for k in stale:
            del self._bridges[k]
        if stale:
            print(
                f"[registry] swept {len(stale)} idle bridge(s); "
                f"{len(self._bridges)} active"
            )


_registry: BridgeRegistry | None = None


def _get_setting(key: str) -> str | None:
    return os.environ.get(key)


def _resolve_port() -> int:
    port = _get_setting("QODER_PORT")
    if not port:
        return 8963
    try:
        return int(port)
    except ValueError:
        print(f"[bridge] WARN invalid QODER_PORT={port!r}; using 8963")
        return 8963


def _extract_pat_from_request(request: Request) -> str | None:
    """从 Authorization: Bearer <PAT> 提取 PAT。"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        if token:
            return token
    return None


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    """随服务启动"每日签到"后台任务（详见 checkin.py）。

    - 未配置 PAT（checkin.json / QODER_CHECKIN_PAT）时任务自动跳过，不影响主链路
    - 服务关闭时取消任务，避免悬挂
    """
    task = checkin.start_background_task(bridge_factory=OpenAiBridge)
    try:
        yield
    finally:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def create_app() -> FastAPI:
    global _registry
    _registry = BridgeRegistry()

    app = FastAPI(title="qoder2api", lifespan=_lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            raw_pat = _extract_pat_from_request(request)
            if not raw_pat:
                return JSONResponse(
                    {
                        "error": {
                            "message": "Missing Authorization: Bearer <PAT>",
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=401,
                )
            real_pat, region = qoder_auth.resolve(raw_pat)
            registry = _registry
            if registry is None:
                raise RuntimeError("bridge registry not initialized")
            bridge = registry.get_or_create(real_pat, region)
            req_body = await request.json()
            return await bridge.handle_chat(req_body)
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status_code=400,
            )
        except Exception as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "qoder_error"}},
                status_code=500,
            )

    @app.get("/v1/models")
    async def list_models(request: Request):
        # 带 PAT 则返回该账户动态拉取的目录; 无 PAT 回退内置表。
        raw_pat = _extract_pat_from_request(request)
        if raw_pat:
            try:
                real_pat, region = qoder_auth.resolve(raw_pat)
                registry = _registry
                if registry is None:
                    raise RuntimeError("bridge registry not initialized")
                bridge = registry.get_or_create(real_pat, region)
                return models.models_payload(await bridge.get_catalog())
            except Exception as e:
                print(f"[models] WARN /v1/models dynamic failed ({e!r}); fallback")
        return models.models_payload()

    return app


def main():
    import uvicorn

    host = _get_setting("QODER_HOST") or "127.0.0.1"
    port = _resolve_port()
    app = create_app()
    print(f"[bridge] listening http://{host}:{port}/v1/chat/completions")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

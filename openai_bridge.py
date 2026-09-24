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
import account_pool as pool_mod
import checkin
import qoder_auth
from qoder_auth import AuthIdentity
import models
import overflow
import reasoning
import usage
from transform import (
    StreamAccumulator,
    ToolCallAccumulator,
    _apply_openai_tool_config,
    _build_qoder_messages,
    _extract_delta,
    _extract_latest_user_prompt,
    _extract_message_images,
    _make_chunk,
    extract_usage_line,
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

    async def handle_chat(self, req_body: dict, on_error=None, on_success=None):
        # on_error/on_success: 账号池故障上报与恢复回调 (pool 模式注入)。
        # 流式在 producer 错误/正常收尾分支调用; 直通模式为 None 保持零行为变化。
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

        # 思考强度: 客户端顶层 reasoning_effort → 上游 parameters 三档注入。
        # 未传/未知值不注入, 保持上游默认 (medium), 与旧行为一致。
        effort = reasoning.apply_reasoning_effort(body, req_body)
        if effort is not None:
            print(f"[bridge] reasoning effort: {req_body.get('reasoning_effort')!r} -> {effort}")

        # 视频适配: 上游不消费视频 part (黑盒实测: image 能看见, video_url/
        # file/顶层 video 全被无视)。最新一条用户消息的 video_url 先抽帧成
        # image parts 再进 prompt 提取 —— 保证 vision 门控和最终请求体看到的是
        # 同一套帧图。ffmpeg 缺失/解码失败在 expand 内部降级为文字提示,
        # 请求照常继续 (不抛错、绝不回退 base64 文本)。
        import video_frames

        messages, vstats = await video_frames.expand_videos_in_messages(messages)
        if vstats:
            print(f"[bridge] video frames: {vstats}")

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
        # 超限翻译上下文: 请求 messages 序列化字符数 (504 归因判据)
        request_chars = len(json.dumps(messages, ensure_ascii=False))

        if stream:
            resp = await self._handle_stream(
                body, url, extra_headers, req_id, created, openai_model, tools_enabled,
                on_error=on_error, on_success=on_success, request_chars=request_chars,
            )
            return resp
        try:
            out = await self._handle_sync(
                body, url, extra_headers, req_id, created, openai_model, tools_enabled
            )
        except Exception as e:
            # 挂载请求体积供路由层 overflow 翻译归因 (504 只有巨体才判溢出)
            try:
                e._qoder_request_chars = request_chars
            except Exception:
                pass
            if on_error is not None:
                on_error(e)
            raise
        if on_success is not None:
            on_success()
        return out

    async def _handle_stream(
        self, body, url, extra_headers, req_id, created, model, tools_enabled,
        on_error=None, on_success=None, request_chars=0,
    ) -> StreamingResponse:
        aq: asyncio.Queue[str | None] = asyncio.Queue()
        acc = StreamAccumulator(
            req_id, created, model, tools_enabled, emit_fn=lambda c: aq.put_nowait(c)
        )

        async def producer():
            collected: usage.UpstreamUsage | None = None
            try:
                async for line in self._open_stream_async(url, body, extra_headers):
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if overflow.stream_error_payload(payload):
                        # 上游在 200 流里塞错误信封 (典型: 413 Range of input length)
                        # —— 旧行为是当空 delta 吞掉, 客户端拿到假"正常空回复";
                        # 现在显式抛错, 走 except 翻译成标准溢出错误。
                        # 截 2000: 413 信封的关键 Range 文本在多层转义后约 300 字符处,
                        # 太短会截掉 Range 导致翻译器丢锚点。
                        raise RuntimeError(f"upstream stream error: {payload[:2000]}")
                    delta = _extract_delta(payload)
                    if not delta.is_empty():
                        acc.accept(delta)
                    found = extract_usage_line(payload)
                    if found is not None:
                        collected = (
                            found if collected is None else collected.merged_with(found)
                        )
                acc.flush()

                # finish chunk
                done = _make_chunk(req_id, created, model)
                done["choices"][0]["finish_reason"] = acc.finish_reason()
                done["choices"][0]["delta"] = {}
                await aq.put(f"data: {json.dumps(done, ensure_ascii=False)}\n\n")

                # usage chunk（OpenAI stream_options.include_usage 风格，必须在 [DONE] 之前）:
                # 下游用 prompt_tokens_details.cached_tokens 算缓存命中率,
                # 用 completion_tokens 算每秒输出 token 数。
                if collected is not None:
                    usage_chunk = _make_chunk(req_id, created, model)
                    usage_chunk["choices"] = []
                    usage_chunk["usage"] = usage.to_openai_usage(collected)
                    await aq.put(
                        f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
                    )
                # 流正常收尾 (客户端中途断连被取消时不会走到这里, 不误报成功)
                if on_success is not None:
                    on_success()
            except Exception as e:
                # 必须把错误暴露给客户端,否则 OpenAI 兼容客户端会把截断的流当成正常结束。
                print(f"[bridge] stream error: {e}")
                if on_error is not None:
                    on_error(e)
                try:
                    err_chunk = _make_chunk(req_id, created, model)
                    err_chunk["choices"][0]["finish_reason"] = "error"
                    err_chunk["choices"][0]["delta"] = {}
                    translated = overflow.translate_upstream_error(str(e), request_chars)
                    if translated is not None:
                        err_chunk["error"] = translated["error"]
                    else:
                        err_chunk["error"] = {
                            "message": str(e),
                            # 上游忙/排队单列一类: 客户端可据此退避重试, 而不是当鉴权失败
                            "type": (
                                "upstream_busy"
                                if isinstance(e, qoder_auth.QoderBusyError)
                                else "qoder_error"
                            ),
                        }
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
        collected: usage.UpstreamUsage | None = None

        async for line in self._open_stream_async(url, body, extra_headers):
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if overflow.stream_error_payload(payload):
                # 上游在 200 流里塞错误信封 (典型: 413 Range of input length)
                # —— 旧行为是当空 delta 吞掉, 客户端拿到假"正常空回复";
                # 现在显式抛错, 走 except 翻译成标准溢出错误。
                # 截 2000: 413 信封的关键 Range 文本在多层转义后约 300 字符处,
                # 太短会截掉 Range 导致翻译器丢锚点。
                raise RuntimeError(f"upstream stream error: {payload[:2000]}")
            delta = _extract_delta(payload)
            if delta.reasoning_content:
                full_reasoning_content.append(delta.reasoning_content)
            if delta.content:
                full_content.append(delta.content)
            if delta.tool_calls and len(delta.tool_calls) > 0:
                tool_calls.append(delta.tool_calls)
            found = extract_usage_line(payload)
            if found is not None:
                collected = found if collected is None else collected.merged_with(found)

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
            # 真实 usage 透传: cached_tokens → 缓存命中率, completion_tokens → 每秒 token 数
            "usage": usage.to_openai_usage(collected),
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

# 多账号池 (单进程单例; 配置来自 pool.json / 环境变量, 详见 pool.py)
_pool: pool_mod.AccountPool | None = None


def get_pool() -> pool_mod.AccountPool | None:
    return _pool


def _get_or_create_bridge(raw_pat: str) -> OpenAiBridge:
    """PAT → Bridge (经共享注册表, 与直通模式复用同一实例)。"""
    real_pat, region = qoder_auth.resolve(raw_pat)
    registry = _registry
    if registry is None:
        raise RuntimeError("bridge registry not initialized")
    bridge = registry.get_or_create(real_pat, region)
    # 昵称回填 (仅首次 bootstrap 后有意义; 幂等, 未 bootstrap 时静默)
    identity = getattr(bridge, "identity", None)
    name = getattr(identity, "name", "") if identity is not None else ""
    p = _pool
    if p is not None and name:
        p.set_label(real_pat, name)
    return bridge


async def _account_for_pat(raw_pat: str) -> OpenAiBridge:
    """取一个账号并保证会话就绪; 成功后顺带把昵称写进池。"""
    bridge = _get_or_create_bridge(raw_pat)
    await bridge.ensure_fresh_session()
    identity = getattr(bridge, "identity", None)
    name = getattr(identity, "name", "") if identity is not None else ""
    p = _pool
    if p is not None and name:
        p.set_label(qoder_auth.resolve(raw_pat)[0], name)
    return bridge


def _pool_report_failure(pat: str, err: Exception) -> None:
    """把请求失败汇报进池状态机 (忙/排队与客户端错误不算账号过错)。

    载荷超限 (HTTP 413)、网关超时 (HTTP 504)、网络超时同样是账号无关的:
    超大请求换任何号都会挂, 记到账号头上会误伤健康池 (曾导致整池空转 WARN)。
    """
    p = _pool
    if p is None:
        return
    if isinstance(err, qoder_auth.QoderBusyError) or isinstance(err, ValueError):
        return
    import httpx

    # httpx 超时/网络中断类异常 str() 常为空串, 只能按类型判 (账号无关)
    if isinstance(err, (httpx.TimeoutException, httpx.TransportError)):
        print(f"[pool] pt-...{pat[-4:]} 网络超时/中断, 不计账号故障")
        return
    msg = str(err)
    if any(m in msg for m in ("HTTP 413", "HTTP 504")):
        print(f"[pool] pt-...{pat[-4:]} 账号无关错误 (超限/网关超时), 不计故障")
        return
    # 巨体请求被网关边缘以 400 弹回 (2026-09-24 实测: 27MB → HTTP 400):
    # 与 413 同类属载荷问题; 小请求的 400 是真参数错误, 照常记账。
    request_chars = getattr(err, "_qoder_request_chars", 0) or 0
    if "HTTP 400" in msg and request_chars >= 1_500_000:
        print(f"[pool] pt-...{pat[-4:]} 载荷过大 (HTTP 400), 不计账号故障")
        return
    p.mark_failure(pat, err)


async def _pool_quota_refresher():
    """周期性巡检账号池余额/配额 (user_status), 把耗尽的号摘出选号池。

    单轮内顺序处理, 每账号失败只记日志不中断整轮; 周期由常量控制。
    """
    import asyncio as _a

    p = _pool
    while p is not None and p.enabled:
        for pat in list(p.pats()):
            try:
                bridge = await _account_for_pat(pat)
                status = await qoder_auth.user_status(
                    bridge.identity.uid,
                    bridge.machine_id,
                    bridge.machine_token,
                    bridge.machine_type,
                    region=bridge.region,
                )
                p.apply_quota_status(pat, status if isinstance(status, dict) else {})
            except _a.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 巡检失败不影响主链路
                print(f"[pool] quota check pt-...{pat[-4:]} failed: {e}")
        # 分片睡眠: 池热加载 (账号增删) 后无需等完整周期即可对新号巡检
        waited = 0.0
        while waited < pool_mod.STATUS_REFRESH_INTERVAL_SEC:
            await _a.sleep(5)
            waited += 5
            if p.reload_if_changed():
                break


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
    project_dir = _get_setting("QODER_PROJECT_DIR") or None
    task = checkin.start_background_task(
        bridge_factory=OpenAiBridge, project_dir=project_dir
    )
    quota_task = None
    if _pool is not None and _pool.enabled:
        quota_task = asyncio.create_task(
            _pool_quota_refresher(), name="qoder-pool-quota-check"
        )
    try:
        yield
    finally:
        for t in (task, quota_task):
            if t is not None and not t.done():
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t


def create_app() -> FastAPI:
    global _registry, _pool
    _registry = BridgeRegistry()
    # 配置目录: 默认与代码同目录; QODER_PROJECT_DIR 可外置 (多实例/容器场景)
    project_dir = _get_setting("QODER_PROJECT_DIR") or os.path.dirname(os.path.abspath(__file__))
    pool_settings = pool_mod.resolve_settings(project_dir=project_dir)
    _pool = pool_mod.AccountPool(pool_settings, project_dir=project_dir)
    if _pool.enabled:
        print(
            f"[pool] 账号池已启用: {len(pool_settings.pats)} 个账号"
            f" (来源 {pool_settings.source or '无'}),"
            f" 网关key={'已配置' if pool_settings.gateway_key else '未配置(仅直通)'}"
        )
    else:
        print("[pool] 未配置账号池 (pool.json/QODER_POOL_PATS), 仅 PAT 直通模式")

    app = FastAPI(title="qoder2api", lifespan=_lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            raw_pat = _extract_pat_from_request(request)
            if not raw_pat:
                return JSONResponse(
                    {
                        "error": {
                            "message": "Missing Authorization: Bearer ***",
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=401,
                )
            req_body = await request.json()
            # 发送前预检 (保守下界): 怎么算都超物理顶的直接回标准溢出错误,
            # 省一次 5 分钟的上游往返; Hermes 分类器认 code 走压缩恢复。
            pre = overflow.pre_reject_error(req_body.get("messages", []))
            if pre is not None:
                return JSONResponse(pre, status_code=400)
            p = _pool
            if p is not None:
                # 热加载: pool.json / checkin.json 被外部改过时增量合并 (2s 节流)
                p.reload_if_changed()

            # ── 双轨鉴权: pt- 前缀直通该账号; 网关 key 走账号池选号 ──
            if raw_pat.startswith("pt-") or not (p is not None and p.gateway_enabled and p.is_gateway_key(raw_pat)):
                bridge = _get_or_create_bridge(raw_pat)
                return await bridge.handle_chat(req_body)

            # 池模式: 粘性选号 → 同步阶段失败自动换号 (排除已试过的, 上限 3 次)
            sticky = pool_mod.derive_sticky_key(req_body)
            tried: set[str] = set()
            last_error: Exception | None = None
            for _ in range(pool_mod.MAX_ATTEMPTS_CAP):
                pat = p.select(sticky, frozenset(tried))
                if pat is None:
                    break
                tried.add(pat)
                try:
                    bridge = await _account_for_pat(pat)
                except Exception as e:  # 冷交换/会话建立失败: 可换号重试
                    _pool_report_failure(pat, e)
                    last_error = e
                    continue
                try:
                    return await bridge.handle_chat(
                        req_body,
                        on_error=lambda err: _pool_report_failure(pat, err),
                        on_success=lambda: p.mark_success(pat, sticky),
                    )
                except (ValueError, qoder_auth.QoderBusyError):
                    # 客户端错误(模型不支持等)与上游忙/排队: 换号无意义, 原样上抛
                    raise
                except Exception as e:
                    # 会话已就绪但请求失败 (auth/网络): 记失败并换下一号
                    _pool_report_failure(pat, e)
                    last_error = e
                    continue
            if last_error is not None:
                raise last_error
            return JSONResponse(
                {
                    "error": {
                        "message": "No healthy account available in pool",
                        "type": "pool_exhausted",
                    }
                },
                status_code=503,
            )
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status_code=400,
            )
        except qoder_auth.QoderBusyError as e:
            # 上游忙/排队: 不是本端的错, 用 503 + Retry-After 让客户端稍后再试
            headers = (
                {"Retry-After": str(e.retry_after_seconds)}
                if e.retry_after_seconds is not None
                else None
            )
            return JSONResponse(
                {"error": {"message": str(e), "type": "upstream_busy"}},
                status_code=503,
                headers=headers,
            )
        except Exception as e:
            # 上游 413/Range/巨体504 → 翻译成 OpenAI context_length_exceeded,
            # Hermes 分类器命中后走压缩恢复而不是重试烧超时
            translated = overflow.translate_upstream_error(
                str(e), getattr(e, "_qoder_request_chars", 0))
            if translated is not None:
                return JSONResponse(translated, status_code=400)
            return JSONResponse(
                {"error": {"message": str(e), "type": "qoder_error"}},
                status_code=500,
            )

    @app.get("/status")
    async def status():
        # 账号池观测端点: 脱敏状态 (无 PAT 明文), 供运维排查与 status.cmd 消费。
        p = _pool
        registry = _registry
        return {
            "pool": {
                "enabled": bool(p is not None and p.enabled),
                "gateway_key_configured": bool(p is not None and p.settings.gateway_key),
                "source": p.settings.source if p is not None else "",
                "accounts": p.snapshot() if p is not None else [],
            },
            "bridges_active": len(registry._bridges) if registry is not None else 0,
        }

    @app.get("/v1/models")
    async def list_models(request: Request):
        # 带 PAT 则返回该账户动态拉取的目录; 无 PAT 回退内置表。
        raw_pat = _extract_pat_from_request(request)
        if raw_pat:
            try:
                p = _pool
                if (
                    p is not None
                    and p.gateway_enabled
                    and not raw_pat.startswith("pt-")
                    and p.is_gateway_key(raw_pat)
                ):
                    # 网关 key: 任选一个健康账号拉目录 (各账号目录一致)
                    pat = p.select("", frozenset())
                    bridge = await _account_for_pat(pat) if pat else None
                else:
                    bridge = _get_or_create_bridge(raw_pat)
                if bridge is not None:
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

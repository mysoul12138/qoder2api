import asyncio
import json
import unittest
from unittest.mock import patch

import models
import transform
import openai_bridge
from fastapi.testclient import TestClient


class PureOpenAiBridgeTests(unittest.TestCase):
    def test_build_messages_uses_only_incoming_openai_messages(self):
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Ok"},
        ]
        converted = transform._build_qoder_messages(messages, "Ok", True)
        self.assertNotIn("Skill", str(converted))

    def test_apply_openai_tool_config_removes_template_tools_when_absent(self):
        body = {
            "tools": [{"type": "function", "function": {"name": "Skill"}}],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        req_body = {"messages": [{"role": "user", "content": "hi"}]}
        tools_enabled = transform._apply_openai_tool_config(body, req_body)
        self.assertFalse(tools_enabled)
        self.assertNotIn("tools", body)
        self.assertNotIn("parallel_tool_calls", body)

    def test_apply_openai_tool_config_keeps_only_request_tools(self):
        template_tool = {"type": "function", "function": {"name": "Skill"}}
        req_tool = {
            "type": "function",
            "function": {"name": "MyTool", "parameters": {}},
        }
        body = {
            "tools": [template_tool],
            "tool_choice": "auto",
        }
        req_body = {
            "tools": [req_tool],
            "tool_choice": "required",
            "messages": [{"role": "user", "content": "hi"}],
        }
        tools_enabled = transform._apply_openai_tool_config(body, req_body)
        self.assertTrue(tools_enabled)
        self.assertEqual(len(body["tools"]), 1)
        self.assertEqual(body["tools"][0]["function"]["name"], "MyTool")
        self.assertEqual(body["tool_choice"], "required")

    def test_tool_history_is_flattened_when_request_tools_are_absent(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "1", "function": {"name": "do", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "content": "result!", "tool_call_id": "1"},
        ]
        converted = transform._build_qoder_messages(messages, "Hi", False)
        self.assertIn("do", converted[1]["content"])

    def test_models_payload_exposes_all_models(self):
        payload = models.models_payload()
        model_ids = {m["id"] for m in payload["data"]}
        self.assertIn("Qwen3.7-Max", model_ids)
        self.assertIn("Qwen3.7-Plus", model_ids)
        for m in payload["data"]:
            self.assertEqual(m["owned_by"], "qoder")

    def test_resolve_model_by_name(self):
        self.assertEqual(models.resolve_model(None), ("Qwen3.7-Max", "qmodel_latest"))
        self.assertEqual(
            models.resolve_model("Qwen3.7-Max"), ("Qwen3.7-Max", "qmodel_latest")
        )

    def test_resolve_model_rejects_unknown(self):
        with self.assertRaises(ValueError):
            models.resolve_model("unknown-model")

    def test_extract_delta_captures_reasoning_content_and_forwards_to_chunk(self):
        line = json.dumps(
            {
                "body": json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": "",
                                    "reasoning_content": "thinking...",
                                }
                            }
                        ],
                    }
                ),
            }
        )
        delta = transform._extract_delta(line)
        self.assertEqual(delta.reasoning_content, "thinking...")
        self.assertFalse(delta.is_empty())

    def test_make_sse_chunk_outputs_reasoning_content_when_present(self):
        chunk = transform._make_sse_chunk(
            "r1", 0, "Qwen3.7-Max", "assistant", "hi", "thinking...", None
        )
        self.assertIn("thinking...", chunk)

    def test_stream_accumulator_forwards_reasoning_and_non_reasoning_deltas(self):
        acc = transform.StreamAccumulator("r1", 0, "m", False)
        acc.accept(transform.BridgeDelta(content="hello"))
        acc.accept(transform.BridgeDelta(reasoning_content="think"))
        acc.accept(transform.BridgeDelta(content=" world"))
        acc.flush()
        result = acc.get_chunks()
        self.assertGreater(len(result), 0)
        self.assertIn("hello", result[0])
        self.assertIn(" world", result[-1])

    def test_models_route_returns_all_models(self):
        client = TestClient(openai_bridge.create_app())
        resp = client.get("/v1/models")
        data = resp.json()["data"]
        model_ids = {m["id"] for m in data}
        self.assertIn("Qwen3.7-Max", model_ids)
        self.assertIn("Qwen3.7-Plus", model_ids)

    def test_chat_route_rejects_unsupported_model(self):
        class DummyBridge:
            def __init__(self, pat, region=None):
                self.pat = pat

            async def handle_chat(self, req_body):
                models.resolve_model(req_body.get("model"))
                return {"ok": True}

        with patch.object(openai_bridge, "OpenAiBridge", DummyBridge):
            client = TestClient(openai_bridge.create_app())
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "unknown-model", "messages": []},
                headers={"Authorization": "Bearer test-pat"},
            )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["type"], "invalid_request_error")

    def test_chat_route_returns_401_without_bearer_token(self):
        client = TestClient(openai_bridge.create_app())
        resp = client.post(
            "/v1/chat/completions", json={"model": "Qwen3.7-Max", "messages": []}
        )
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.json()["error"]["type"], "invalid_request_error")

    def test_extract_message_images_and_build_user_message(self):
        data_url = "data:image/png;base64,iVBORw0KGgo="
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "input_image", "image_url": {"url": "https://x/a.jpg"}},
            ],
        }
        urls = transform._extract_message_images(msg)
        self.assertEqual(urls, [data_url, "https://x/a.jpg"])

        built = transform._build_user_message("what is this?", urls)
        contents = built["contents"]
        # images first, text last
        self.assertEqual(
            contents[0], {"type": "image_url", "image_url": {"url": data_url}}
        )
        self.assertEqual(
            contents[1], {"type": "image_url", "image_url": {"url": "https://x/a.jpg"}}
        )
        self.assertEqual(contents[-1], {"type": "text", "text": "what is this?"})

    def test_build_user_message_image_only(self):
        # a user message with only an image (no text) should still be built
        built = transform._build_user_message("", ["data:image/png;base64,AAA"])
        types = [p["type"] for p in built["contents"]]
        self.assertEqual(types, ["image_url"])

    def test_convert_incoming_user_message_with_image(self):
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAA"},
                },
                {"type": "text", "text": "describe"},
            ],
        }
        out = transform._convert_incoming_message(
            msg, tools_enabled=False, allow_structured_tool_calls=False
        )
        if out is None:
            self.fail("_convert_incoming_message returned None")
        self.assertEqual(out["role"], "user")
        self.assertEqual(out["contents"][0]["type"], "image_url")
        self.assertEqual(out["contents"][-1]["text"], "describe")

    def test_vision_models_derived_from_catalog(self):
        # 视觉能力来自 catalog 的 is_vl 字段 (动态), 不再写死常量
        raw = {
            "chat": [
                {"key": "a", "display_name": "ModelA", "enable": True, "is_vl": True},
                {"key": "b", "display_name": "ModelB", "enable": True, "is_vl": False},
                {"key": "auto", "display_name": "Auto", "enable": True, "is_vl": True},
            ]
        }
        cat = models.extract_catalog(raw)
        assert cat is not None
        self.assertIn("ModelA", cat.vision_models)
        self.assertNotIn("ModelB", cat.vision_models)
        self.assertNotIn("Auto", cat.model_map)  # auto 是路由入口, 被跳过
        # 默认兜底目录: MiniMax (is_vl=false) 不支持视觉, 其余支持
        default = models.default_catalog()
        self.assertNotIn("MiniMax-M2.7", default.vision_models)
        self.assertIn("Qwen3.7-Plus", default.vision_models)

    def test_chat_route_rejects_image_with_non_vision_model(self):
        # MiniMax-M2.7 在 catalog 里 is_vl=false, 应拒绝图片输入
        class DummyBridge:
            def __init__(self, pat, region=None):
                self.pat = pat

            def get_catalog(self):
                return models.default_catalog()

            async def handle_chat(self, req_body):
                # mirror real guard: images + non-vision model → ValueError
                catalog = self.get_catalog()
                messages = req_body.get("messages", [])
                has_images = any(
                    transform._extract_message_images(m)
                    for m in messages
                    if isinstance(m, dict)
                )
                model, _ = models.resolve_model(req_body.get("model"), catalog)
                if has_images and model not in catalog.vision_models:
                    raise ValueError(
                        f"Image input is not supported by model '{model}'."
                    )
                return {"ok": True}

        with patch.object(openai_bridge, "OpenAiBridge", DummyBridge):
            client = TestClient(openai_bridge.create_app())
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "MiniMax-M2.7",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,AAA"},
                                },
                                {"type": "text", "text": "describe"},
                            ],
                        }
                    ],
                },
                headers={"Authorization": "Bearer test-pat"},
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not supported", resp.json()["error"]["message"])

    def test_stream_error_emits_error_chunk_before_done(self):
        """reader 线程异常时应该先发一个 finish_reason='error' 的 chunk,再发 [DONE]。

        避免 OpenAI 兼容客户端把截断的流误当成正常结束。"""
        with patch.object(openai_bridge, "OpenAiBridge", _ErroringBridge):
            client = TestClient(openai_bridge.create_app())
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "Qwen3.7-Max",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer test-pat"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.text
        # 错误 chunk 必须出现在 [DONE] 之前
        done_idx = body.find("data: [DONE]")
        self.assertGreater(done_idx, -1, "missing [DONE] sentinel")
        prefix = body[:done_idx]
        self.assertIn('"finish_reason": "error"', prefix)
        self.assertIn('"error"', prefix)
        self.assertIn("boom", prefix)

    # ── 回归测试: review 修复 (#1/#3/#4/#5) ───────────────────────────

    async def _drain(self, gen):
        """把异步生成器的产出收集成 list (供 _open_stream_async 测试)。"""
        out = []
        async for x in gen:
            out.append(x)
        return out

    async def _run_concurrent(self, *aws):
        """并发等待多个 awaitable (gather 必须在事件循环内调用)。"""
        await asyncio.gather(*aws)

    def test_refresh_lock_is_asyncio_lock(self):
        """#1: 续期锁必须是 asyncio.Lock —— threading.Lock 跨 await 会
        阻塞整个事件循环 (其他 PAT 的请求也会被冻结)。"""
        bridge = openai_bridge.OpenAiBridge("pt-x")
        self.assertIsInstance(bridge._refresh_lock, asyncio.Lock)

    def test_bootstrap_session_is_deduplicated_under_concurrency(self):
        """#3: 同一 Bridge 并发首请求只做一次冷交换 (asyncio.Lock + 双重检查)。"""
        bridge = openai_bridge.OpenAiBridge("pt-x")
        jt = {
            "name": "u",
            "id": "1",
            "userType": "personal_standard",
            "securityOauthToken": "s",
            "refreshToken": "r",
            "expireTime": 9999999999999,
        }
        exchanges = {"n": 0}

        async def fake_exchange(*a, **k):
            exchanges["n"] += 1
            await asyncio.sleep(0.01)  # 放大竞态窗口
            return jt

        with patch.object(
            openai_bridge.qoder_auth, "exchange_job_token", fake_exchange
        ):
            asyncio.run(
                self._run_concurrent(
                    bridge.ensure_fresh_session(),
                    bridge.ensure_fresh_session(),
                    bridge.ensure_fresh_session(),
                )
            )
        self.assertEqual(exchanges["n"], 1)
        self.assertTrue(bridge._bootstrapped)
        self.assertIsNotNone(bridge.sess)

    def test_open_stream_no_retry_after_content_produced(self):
        """#4: 已产出内容行后遇到 401, 不再重试 (否则上游重发 prompt 导致重复),
        直接抛错让 _handle_stream 走错误分支。"""
        bridge = openai_bridge.OpenAiBridge("pt-x")
        bridge.sess = object()  # type: ignore[assignment]  # open_stream_lines 被 mock, sess 值无关
        refreshed = {"n": 0}

        async def fake_refresh():
            refreshed["n"] += 1

        bridge._force_refresh = fake_refresh
        call = {"n": 0}

        async def fake_stream(*a, **k):
            call["n"] += 1
            yield "data:real-content"  # 先产出一行真实内容
            raise openai_bridge.qoder_auth.QoderAuthError(401, "mid-stream")

        with (
            patch.object(openai_bridge.qoder_auth, "open_stream_lines", fake_stream),
            self.assertRaises(openai_bridge.qoder_auth.QoderAuthError),
        ):
            asyncio.run(self._drain(bridge._open_stream_async("u", {}, None)))
        self.assertEqual(call["n"], 1)  # 上游只调用一次, 没重试
        self.assertEqual(refreshed["n"], 0)  # 没刷新

    def test_open_stream_retries_when_no_content_yet(self):
        """#4 反向: 尚未产出任何内容时遇到 401, 刷新并重试一次后成功。"""
        bridge = openai_bridge.OpenAiBridge("pt-x")
        bridge.sess = object()  # type: ignore[assignment]
        refreshed = {"n": 0}

        async def fake_refresh():
            refreshed["n"] += 1

        bridge._force_refresh = fake_refresh
        call = {"n": 0}

        async def fake_stream(*a, **k):
            call["n"] += 1
            if call["n"] == 1:
                raise openai_bridge.qoder_auth.QoderAuthError(401, "early")
            yield "data:after-refresh"

        with patch.object(openai_bridge.qoder_auth, "open_stream_lines", fake_stream):
            out = asyncio.run(self._drain(bridge._open_stream_async("u", {}, None)))
        self.assertEqual(call["n"], 2)  # 重试了一次
        self.assertEqual(refreshed["n"], 1)  # 刷新了一次
        self.assertEqual(out, ["data:after-refresh"])

    def test_registry_enforces_hard_cap(self):
        """#5: 注册表条目数任何时候都不超过硬上限。"""
        openai_bridge._BRIDGE_MAX_ENTRIES = 3
        try:
            reg = openai_bridge.BridgeRegistry()
            region = openai_bridge.qoder_auth.CN
            for i in range(6):
                reg.get_or_create(f"pt-{i}", region)
            self.assertEqual(len(reg._bridges), 3)
        finally:
            openai_bridge._BRIDGE_MAX_ENTRIES = 1024

    def test_registry_evicts_idle_entries(self):
        """#5: 空闲超过 TTL 的条目在惰性 sweep 时被清除。"""
        openai_bridge._BRIDGE_TTL_SEC = 0.1
        openai_bridge._BRIDGE_SWEEP_EVERY = 1
        try:
            reg = openai_bridge.BridgeRegistry()
            region = openai_bridge.qoder_auth.CN
            reg.get_or_create("pt-a", region)
            # 把访问时间调到 TTL 之前 (避免真实 sleep, 测试更快且不依赖墙钟)
            for e in reg._bridges.values():
                e.last_access -= 1.0
            reg.get_or_create("pt-b", region)  # 新建触发 sweep
            self.assertEqual(len(reg._bridges), 1)  # pt-a 已过期被清, 只剩 pt-b
        finally:
            openai_bridge._BRIDGE_TTL_SEC = 30 * 60
            openai_bridge._BRIDGE_SWEEP_EVERY = 64


def transform_identity_stub():
    return openai_bridge.AuthIdentity(
        name="t",
        aid="t",
        uid="t",
        yx_uid="",
        organization_id="",
        organization_name="",
        user_type="personal_standard",
        security_oauth_token="",
        refresh_token="",
    )


def _minimal_template_stub():
    # baseprompt.json 在 handle_chat 中被 copy.deepcopy + 赋值的最小结构,
    # 与 _build_qoder_messages 的调用路径保持一致。
    return {
        "model_config": {"key": "", "source": "system", "is_reasoning": False},
        "chat_context": {
            "text": {"text": ""},
            "extra": {
                "originalContent": {"text": ""},
                "modelConfig": {"key": "", "is_reasoning": False},
            },
        },
        "business": {"id": "", "begin_at": 0, "name": ""},
        "messages": [],
    }


class _ErroringBridge(openai_bridge.OpenAiBridge):
    """流式 reader 抛错的 Bridge 替身 (绕过真实 PAT→jobToken 交换)。

    提到模块级以避免在测试方法内嵌套含 yield 的生成器 (会误触
    no-return-value-in-generator 规则)。
    """

    def __init__(self, pat, region=None):
        self._pat = pat
        self.region = region or openai_bridge.qoder_auth.CN
        self.identity = transform_identity_stub()
        self.template_base = _minimal_template_stub()
        self._catalog = None
        self._catalog_ts = 0.0
        self._catalog_lock = openai_bridge.threading.Lock()
        self._bootstrapped = True

    async def ensure_fresh_session(self):
        return

    async def get_catalog(self):
        return models.default_catalog()

    async def _open_stream_async(self, url, body, extra_headers):
        raise RuntimeError("boom")
        yield  # 使其成为异步生成器


if __name__ == "__main__":
    unittest.main()

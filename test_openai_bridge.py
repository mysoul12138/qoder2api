import copy
import json
import unittest
from unittest.mock import patch

import models
import transform
import openai_bridge


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
        app = openai_bridge.create_app()
        resp = app.test_client().get("/v1/models")
        data = resp.get_json()["data"]
        model_ids = {m["id"] for m in data}
        self.assertIn("Qwen3.7-Max", model_ids)
        self.assertIn("Qwen3.7-Plus", model_ids)

    def test_chat_route_rejects_unsupported_model(self):
        class DummyBridge:
            def __init__(self, pat, region=None):
                self.pat = pat

            def handle_chat(self, req_body):
                models.resolve_model(req_body.get("model"))
                return {"ok": True}

        with patch.object(openai_bridge, "OpenAiBridge", DummyBridge):
            app = openai_bridge.create_app()
            resp = app.test_client().post(
                "/v1/chat/completions",
                json={"model": "unknown-model", "messages": []},
                headers={"Authorization": "Bearer test-pat"},
            )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"]["type"], "invalid_request_error")

    def test_chat_route_returns_401_without_bearer_token(self):
        app = openai_bridge.create_app()
        resp = app.test_client().post(
            "/v1/chat/completions", json={"model": "Qwen3.7-Max", "messages": []}
        )
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.get_json()["error"]["type"], "invalid_request_error")

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

            def handle_chat(self, req_body):
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
            app = openai_bridge.create_app()
            resp = app.test_client().post(
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
        self.assertIn("not supported", resp.get_json()["error"]["message"])

    def test_stream_error_emits_error_chunk_before_done(self):
        """reader 线程异常时应该先发一个 finish_reason='error' 的 chunk,再发 [DONE]。

        避免 OpenAI 兼容客户端把截断的流误当成正常结束。"""

        class ErroringBridge(openai_bridge.OpenAiBridge):
            def __init__(self, pat, region=None):
                # 绕过真实的 PAT→jobToken 交换,手工填上流式路径需要的最小字段。
                self._pat = pat
                self.region = region or openai_bridge.qoder_auth.CN
                self.identity = transform_identity_stub()
                self.template_base = _minimal_template_stub()
                # get_catalog 需要的目录缓存字段
                self._catalog = None
                self._catalog_ts = 0.0
                self._catalog_lock = openai_bridge.threading.Lock()

            def ensure_fresh_session(self):
                return

            def _open_stream_with_retry(self, url, body, extra_headers, on_line):
                raise RuntimeError("boom")

        with patch.object(openai_bridge, "OpenAiBridge", ErroringBridge):
            app = openai_bridge.create_app()
            resp = app.test_client().post(
                "/v1/chat/completions",
                json={
                    "model": "Qwen3.7-Max",
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer test-pat"},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.get_data(as_text=True)
        # 错误 chunk 必须出现在 [DONE] 之前
        done_idx = body.find("data: [DONE]")
        self.assertGreater(done_idx, -1, "missing [DONE] sentinel")
        prefix = body[:done_idx]
        self.assertIn('"finish_reason": "error"', prefix)
        self.assertIn('"error"', prefix)
        self.assertIn("boom", prefix)


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


if __name__ == "__main__":
    unittest.main()

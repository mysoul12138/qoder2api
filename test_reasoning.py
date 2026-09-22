"""reasoning 档位映射单元测试 + handle_chat 注入接线测试 (全离线)。"""

import copy
import unittest
from unittest.mock import patch

import openai_bridge
import reasoning
import models


class NormalizeTests(unittest.TestCase):
    def test_wire_values_map_to_upstream_three_tiers(self):
        cases = {
            "none": "low", "minimal": "low", "low": "low",
            "medium": "medium", "high": "medium",
            "xhigh": "xhigh", "max": "xhigh", "ultra": "xhigh",
            "MAX": "xhigh", "  low  ": "low",  # 大小写与空白鲁棒
        }
        for wire, want in cases.items():
            self.assertEqual(reasoning.normalize_effort(wire), want, wire)

    def test_unknown_values_fall_back_to_no_injection(self):
        for raw in (None, "", "banana", 5, True, [], {}):
            self.assertIsNone(reasoning.normalize_effort(raw), repr(raw))


class ApplyTests(unittest.TestCase):
    def _body(self):
        return {"model_config": {"key": "qmodel_latest"}, "business": {}}

    def test_injects_parameters_dict(self):
        body = self._body()
        effort = reasoning.apply_reasoning_effort(body, {"reasoning_effort": "max"})
        self.assertEqual(effort, "xhigh")
        self.assertEqual(
            body["parameters"], {"enable_thinking": True, "reasoning_effort": "xhigh"}
        )

    def test_absent_effort_leaves_body_untouched(self):
        body = self._body()
        snapshot = copy.deepcopy(body)
        self.assertIsNone(reasoning.apply_reasoning_effort(body, {"messages": []}))
        self.assertEqual(body, snapshot, "未传档位不得改动请求体 (保持上游默认 medium)")

    def test_hostile_template_parameters_replaced_not_crashed(self):
        body = self._body()
        body["parameters"] = "garbage-not-a-dict"  # 模板被上游改坏时防御
        self.assertEqual(
            reasoning.apply_reasoning_effort(body, {"reasoning_effort": "low"}), "low"
        )
        self.assertEqual(body["parameters"]["reasoning_effort"], "low")

    def test_existing_parameters_keys_preserved(self):
        body = self._body()
        body["parameters"] = {"temperature": 0.7}
        reasoning.apply_reasoning_effort(body, {"reasoning_effort": "medium"})
        self.assertEqual(body["parameters"]["temperature"], 0.7, "不丢模板既有参数")


class HandleChatWiringTests(unittest.TestCase):
    """真实 handle_chat 路径: 请求体带 reasoning_effort 时上游 body 收到 parameters。"""

    class _StubBridge:
        """绕开网络的最小桥: 只提供 handle_chat 依赖的会话/目录, 捕获上游 body。"""

        def __init__(self, req_body_capture):
            self._cap = req_body_capture
            self.region = openai_bridge.qoder_auth.CN
            self.identity = openai_bridge.AuthIdentity(
                name="t", aid="a", uid="u", yx_uid="", organization_id="",
                organization_name="", user_type="personal_standard",
                security_oauth_token="", refresh_token="",
            )
            self.template_base = {
                "model_config": {"key": "", "is_reasoning": False},
                "chat_context": {
                    "extra": {
                        "modelConfig": {"key": "", "is_reasoning": False},
                        "originalContent": {"text": ""},
                    },
                    "text": {"text": ""},
                },
                "business": {"id": "", "name": "", "begin_at": 0},
                "messages": [],
            }

        async def ensure_fresh_session(self):
            return None

        async def get_catalog(self):
            return models.default_catalog()

        async def _handle_stream(self, body, *a, **k):
            self._cap["body"] = body
            return {"streamed": True}

        async def _handle_sync(self, body, *a, **k):
            self._cap["body"] = body
            return {"synced": True}

    def _run(self, req_body):
        import asyncio

        cap = {}
        bridge = openai_bridge.OpenAiBridge("pt-fake")
        stub = self._StubBridge(cap)
        bridge.identity = stub.identity
        bridge.template_base = copy.deepcopy(stub.template_base)
        with patch.object(openai_bridge.OpenAiBridge, "ensure_fresh_session",
                          lambda self: stub.ensure_fresh_session()), \
             patch.object(openai_bridge.OpenAiBridge, "get_catalog",
                          lambda self: stub.get_catalog()), \
             patch.object(openai_bridge.OpenAiBridge, "_handle_sync",
                          stub._handle_sync):
            req = dict(req_body)
            req.setdefault("model", "Qwen3.7-Max")
            req.setdefault("messages", [{"role": "user", "content": "hi"}])
            asyncio.run(bridge.handle_chat(req))
        return cap["body"]

    def test_max_maps_to_xhigh_on_wire_body(self):
        body = self._run({"reasoning_effort": "max"})
        self.assertEqual(body["parameters"],
                         {"enable_thinking": True, "reasoning_effort": "xhigh"})

    def test_no_effort_no_parameters_on_wire_body(self):
        body = self._run({})
        self.assertNotIn("parameters", body)


if __name__ == "__main__":
    unittest.main()

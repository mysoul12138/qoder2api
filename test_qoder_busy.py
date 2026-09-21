"""上游"忙 / 排队"（非鉴权）分类 + "不刷新会话"回归测试。

背景: 上游会用 HTTP 200 + 信封 {"statusCodeValue":403,"body":"{...code 10605...}"}
下发忙/排队信号; 旧实现把它当成鉴权失败 → 白刷新一次会话（token 轮换还会连累
其他在途请求）→ 重试后依旧失败 → 最终 500。样本来自 2026-09-21 实测日志。
"""

import asyncio
import json
import unittest
from unittest.mock import patch

import openai_bridge
import qoder_auth
from fastapi.testclient import TestClient


BUSY_ENVELOPE = {
    "headers": {"Content-Type": ["application/json"]},
    "body": json.dumps(
        {
            "code": "10605",
            "message": json.dumps(
                {
                    "isQueued": False,
                    "modelKey": "q37fmodel",
                    "queueCount": 0,
                    "queueType": "p3",
                    "retryAfterSeconds": 5,
                    "serviceAvailable": True,
                    "waitTime": 0,
                }
            ),
        }
    ),
    "statusCodeValue": 403,
    "statusCode": "FORBIDDEN",
}

AUTH_ENVELOPE = {
    "headers": {"Content-Type": ["application/json"]},
    "body": json.dumps({"code": "105", "message": "Login expired"}),
    "statusCodeValue": 403,
    "statusCode": "FORBIDDEN",
}


def _line(obj) -> str:
    return "data:" + json.dumps(obj, ensure_ascii=False)


async def _drain(agen) -> list:
    return [item async for item in agen]


class BusyClassificationTests(unittest.TestCase):
    def test_busy_envelope_is_not_auth(self):
        is_auth, _ = qoder_auth._detect_in_stream_auth_error(_line(BUSY_ENVELOPE))
        self.assertFalse(is_auth, "忙/排队信号不该被判成鉴权失败")

    def test_busy_envelope_detected_with_retry_after(self):
        is_busy, detail, retry_after = qoder_auth.detect_upstream_busy(BUSY_ENVELOPE)
        self.assertTrue(is_busy)
        self.assertEqual(retry_after, 5)
        self.assertIn("10605", detail)

    def test_busy_line_helper_matches(self):
        self.assertTrue(qoder_auth._detect_busy_line(_line(BUSY_ENVELOPE))[0])
        self.assertFalse(qoder_auth._detect_busy_line(_line(AUTH_ENVELOPE))[0])

    def test_auth_envelope_still_classified_as_auth(self):
        is_auth, detail = qoder_auth._detect_in_stream_auth_error(_line(AUTH_ENVELOPE))
        self.assertTrue(is_auth, "真正的登录过期(105)必须仍然被判成鉴权失败")
        self.assertFalse(qoder_auth.detect_upstream_busy(AUTH_ENVELOPE)[0])

    def test_plain_401_is_auth_not_busy(self):
        envelope = {"body": "", "statusCodeValue": 401, "statusCode": "UNAUTHORIZED"}
        self.assertTrue(qoder_auth._detect_in_stream_auth_error(_line(envelope))[0])
        self.assertFalse(qoder_auth.detect_upstream_busy(envelope)[0])

    def test_bare_body_without_envelope(self):
        # 没有信封包裹的裸 body 也要能识别（非 200 响应体常见形态）
        obj = {"code": "10605", "message": json.dumps({"isQueued": True, "waitTime": 12})}
        is_busy, _, retry_after = qoder_auth.detect_upstream_busy(obj)
        self.assertTrue(is_busy)
        self.assertIsNone(retry_after)

    def test_markers_without_code(self):
        obj = {"message": json.dumps({"isQueued": True, "retryAfterSeconds": 30})}
        is_busy, _, retry_after = qoder_auth.detect_upstream_busy(obj)
        self.assertTrue(is_busy)
        self.assertEqual(retry_after, 30)

    def test_ordinary_content_line_is_neither(self):
        line = _line({"body": json.dumps({"choices": [{"delta": {"content": "hi"}}]})})
        self.assertFalse(qoder_auth._detect_in_stream_auth_error(line)[0])
        self.assertFalse(qoder_auth._detect_busy_line(line)[0])


class BusyDoesNotRefreshSessionTests(unittest.TestCase):
    """核心回归: 忙/排队时必须"不刷新会话"。

    旧实现会 force-refresh 一次（白轮换 token）再重试。"""

    def test_busy_error_does_not_trigger_refresh(self):
        bridge = openai_bridge.OpenAiBridge("pt-x")
        bridge.sess = object()  # open_stream_lines 被 mock, sess 值无关
        refreshed = {"n": 0}

        async def fake_refresh():
            refreshed["n"] += 1

        async def fake_stream(sess, url, body, headers):
            raise qoder_auth.QoderBusyError("10605 queued", 5)
            yield  # pragma: no cover

        with (
            patch.object(openai_bridge.qoder_auth, "open_stream_lines", fake_stream),
            patch.object(bridge, "_force_refresh", fake_refresh),
            self.assertRaises(qoder_auth.QoderBusyError),
        ):
            asyncio.run(_drain(bridge._open_stream_async("u", {}, None)))
        self.assertEqual(refreshed["n"], 0, "忙/排队时不该刷新会话")

    def test_auth_error_still_refreshes_once(self):
        # 对照: 真正的鉴权失败仍要刷新一次（保留既有行为）
        bridge = openai_bridge.OpenAiBridge("pt-x")
        bridge.sess = object()
        refreshed = {"n": 0}
        calls = {"n": 0}

        async def fake_refresh():
            refreshed["n"] += 1

        async def fake_stream(sess, url, body, headers):
            calls["n"] += 1
            if calls["n"] == 1:
                raise qoder_auth.QoderAuthError(401, "early")
            yield "data:after-refresh"

        with (
            patch.object(openai_bridge.qoder_auth, "open_stream_lines", fake_stream),
            patch.object(bridge, "_force_refresh", fake_refresh),
        ):
            out = asyncio.run(_drain(bridge._open_stream_async("u", {}, None)))
        self.assertEqual(refreshed["n"], 1)
        self.assertEqual(out, ["data:after-refresh"])


class _BusyRouteBridge(openai_bridge.OpenAiBridge):
    """handle_chat 直接抛忙错误，用于验证路由映射。"""

    async def handle_chat(self, req_body):
        raise qoder_auth.QoderBusyError("10605 queued", 5)


class BusyRouteTests(unittest.TestCase):
    def test_route_returns_503_with_retry_after(self):
        with patch.object(openai_bridge, "OpenAiBridge", _BusyRouteBridge):
            client = TestClient(openai_bridge.create_app())
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "Qwen3.7-Max",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer test-pat"},
            )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.headers.get("retry-after"), "5")
        self.assertEqual(resp.json()["error"]["type"], "upstream_busy")


class _BusyStreamBridge(openai_bridge.OpenAiBridge):
    """_open_stream_async 直接抛忙错误，用于验证流内错误 chunk 的类型。"""

    def __init__(self):
        super().__init__("pt-test")

    async def _open_stream_async(self, url, body, extra_headers):
        raise qoder_auth.QoderBusyError("10605 queued", 5)
        yield  # pragma: no cover


class BusyStreamChunkTests(unittest.TestCase):
    def test_stream_error_chunk_type_is_upstream_busy(self):
        bridge = _BusyStreamBridge()

        async def run():
            resp = await bridge._handle_stream({}, "u", None, "chatcmpl-t", 1, "m", False)
            chunks = []
            async for chunk in resp.body_iterator:
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
            return chunks

        body = "".join(asyncio.run(run()))
        err_lines = [
            json.loads(line[len("data:"):])
            for line in body.split("\n")
            if line.startswith("data:") and '"error"' in line
        ]
        self.assertEqual(len(err_lines), 1)
        self.assertEqual(err_lines[0]["error"]["type"], "upstream_busy")
        self.assertEqual(err_lines[0]["choices"][0]["finish_reason"], "error")
        self.assertIn("data: [DONE]", body)


if __name__ == "__main__":
    unittest.main()

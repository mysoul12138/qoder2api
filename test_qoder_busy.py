"""上游"忙 / 排队"（非鉴权）分类 + "不刷新会话"回归测试。

背景: 上游会用 HTTP 200 + 信封 {"statusCodeValue":403,"body":"{...code 10605...}"}
下发忙/排队信号; 旧实现把它当成鉴权失败 → 白刷新一次会话（token 轮换还会连累
其他在途请求）→ 重试后依旧失败 → 最终 500。样本来自 2026-09-21 实测日志。
"""

import asyncio
import json
import re
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

# 实测(2026-09-21)的"双层信封"形态: 外层 code 403 的 message 里再套一层 code 10605
# 的信封, 队列详情（含 retryAfterSeconds）埋在第二层的 message 里。只挖一层的旧实现
# 解析不到 retryAfterSeconds, 503 就丢掉了 Retry-After。
NESTED_BUSY_ENVELOPE = {
    "headers": {"Content-Type": ["application/json"]},
    "body": json.dumps(
        {
            "code": "403",
            "message": json.dumps(
                {
                    "code": "10605",
                    "message": json.dumps(
                        {
                            "isQueued": True,
                            "modelKey": "qfmodel",
                            "queueCount": 8079,
                            "queueType": "p3",
                            "retryAfterSeconds": 30,
                            "serviceAvailable": True,
                            "waitTime": 393,
                        }
                    ),
                }
            ),
        }
    ),
    "statusCodeValue": 403,
    "statusCode": "FORBIDDEN",
}

BARE_NESTED_BUSY = {
    "code": "403",
    "message": json.dumps(
        {
            "code": "10605",
            "message": json.dumps(
                {
                    "isQueued": True,
                    "modelKey": "qfmodel",
                    "queueCount": 8079,
                    "queueType": "p3",
                    "retryAfterSeconds": 30,
                    "serviceAvailable": True,
                    "waitTime": 393,
                }
            ),
        }
    ),
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


class NestedBusyEnvelopeTests(unittest.TestCase):
    """双层信封（外层 403 套内层 10605）必须下钻到队列详情层。"""

    def test_nested_envelope_drills_to_retry_after(self):
        is_busy, detail, retry_after = qoder_auth.detect_upstream_busy(NESTED_BUSY_ENVELOPE)
        self.assertTrue(is_busy)
        self.assertEqual(retry_after, 30, "retryAfterSeconds 在第二层 message 里, 必须挖到")
        self.assertIn("10605", detail)
        self.assertIn("queueCount", detail)

    def test_nested_bare_body_drills_to_retry_after(self):
        is_busy, _, retry_after = qoder_auth.detect_upstream_busy(BARE_NESTED_BUSY)
        self.assertTrue(is_busy)
        self.assertEqual(retry_after, 30)

    def test_nested_busy_line_detected_in_stream(self):
        self.assertTrue(qoder_auth._detect_busy_line(_line(NESTED_BUSY_ENVELOPE))[0])

    def test_nested_envelope_is_not_auth(self):
        is_auth, _ = qoder_auth._detect_in_stream_auth_error(_line(NESTED_BUSY_ENVELOPE))
        self.assertFalse(is_auth, "双层信封的忙信号同样不该被判成鉴权失败")

    def test_depth_limit_fails_safe(self):
        # 超过下钻上限的异常嵌套: 不抛异常、不硬判成忙（保持未知形态的既有语义）
        deep = {"isQueued": True, "retryAfterSeconds": 30}
        for _ in range(qoder_auth._BUSY_DRILL_LIMIT + 2):
            deep = {"code": "403", "message": json.dumps(deep)}
        is_busy, _, retry_after = qoder_auth.detect_upstream_busy(deep)
        self.assertFalse(is_busy)
        self.assertIsNone(retry_after)


class BusyMessageTextTests(unittest.TestCase):
    """忙错误的消息文本刻意不含可被解析的"重试时间"短语。

    桌面端(Hermes)会把错误里解析出的任何重试时间渲染成"限额将于 X 重置"的
    倒计时（把排队误当成额度重置），所以消息里不写 "(retry after Ns)" ——
    这里用 Hermes 的同款正则做防回归（hermes agent/retry_utils.py）。
    """

    # 与 Hermes agent/retry_utils.py::_RETRY_AFTER_SECONDS_RE 同款
    _RETRY_PHRASE_RE = re.compile(
        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
        re.IGNORECASE,
    )

    def test_message_has_no_parseable_retry_phrase(self):
        err = qoder_auth.QoderBusyError("10605 queued", 30)
        self.assertIsNone(self._RETRY_PHRASE_RE.search(str(err)))

    def test_real_busy_detail_text_does_not_parse(self):
        # 真实忙信号解出的详情里带 "retryAfterSeconds": 30 —— 它同样不能被匹配到
        is_busy, detail, retry_after = qoder_auth.detect_upstream_busy(NESTED_BUSY_ENVELOPE)
        self.assertTrue(is_busy)
        msg = str(qoder_auth.QoderBusyError(detail, retry_after))
        self.assertIsNone(self._RETRY_PHRASE_RE.search(msg))

    def test_retry_seconds_still_ride_on_the_attribute(self):
        # 头部映射依赖这个属性（非流式 503 的 Retry-After），不能被一起删掉
        err = qoder_auth.QoderBusyError("10605 queued", 30)
        self.assertEqual(err.retry_after_seconds, 30)
        self.assertIn("10605", str(err))


class _NestedBusyRouteBridge(openai_bridge.OpenAiBridge):
    """handle_chat 抛"双层信封"解出的忙错误, 用于验证 Retry-After 头来自最里层。"""

    async def handle_chat(self, req_body):
        is_busy, detail, retry_after = qoder_auth.detect_upstream_busy(NESTED_BUSY_ENVELOPE)
        assert is_busy
        raise qoder_auth.QoderBusyError(detail, retry_after)


class NestedBusyRouteTests(unittest.TestCase):
    def test_route_retry_after_comes_from_nested_envelope(self):
        with patch.object(openai_bridge, "OpenAiBridge", _NestedBusyRouteBridge):
            client = TestClient(openai_bridge.create_app())
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "Qwen3.8-Flash",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer test-pat"},
            )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.headers.get("retry-after"), "30")
        self.assertEqual(resp.json()["error"]["type"], "upstream_busy")


if __name__ == "__main__":
    unittest.main()

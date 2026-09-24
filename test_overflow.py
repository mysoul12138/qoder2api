"""overflow 翻译与预检单元测试 + handle_chat 接线 (全离线)。"""

import json
import unittest

import overflow


class TranslateTests(unittest.TestCase):
    def test_range_413_translates_to_context_overflow(self):
        # 真实上游原文 (2026-09-24 视频请求现场)
        raw = (
            'HTTP 413 {"headers":{"Content-Type":["application/json"]},"body":"{'
            '\\"code\\":\\"provider_error\\",\\"details\\":\\"{\\\\\\"error\\\\\\":'
            '{\\\\\\"message\\\\\\":\\\\\\"<400> InternalError.Algo.InvalidParameter: '
            'Range of input length should be [1, 983616]\\\\\\"}\\""}\''
        )
        err = overflow.translate_upstream_error(raw, 3_488_006)
        self.assertIsNotNone(err)
        e = err["error"]
        self.assertEqual(e["code"], "context_length_exceeded")
        # Hermes 分类器双命中: message 含 maximum context length, code 在 code 表
        self.assertIn("maximum context length", e["message"])
        self.assertIn("983616", e["message"])

    def test_hermes_classifier_would_hit_this_shape(self):
        # 守护: 复刻 Hermes _CONTEXT_OVERFLOW_PATTERNS 的关键子串, 消息形状变了要察觉
        e = overflow.make_overflow_error(1, 983616, "x")["error"]
        msg = e["message"].lower()
        self.assertTrue(any(p in msg for p in
                            ("maximum context length", "context length", "context_length_exceeded")))
        self.assertEqual(e["code"], "context_length_exceeded")

    def test_payload_too_large_aliases(self):
        for raw in ("HTTP 413 PAYLOAD_TOO_LARGE", "error code: 413 something"):
            err = overflow.translate_upstream_error(raw, 100)
            self.assertIsNotNone(err, raw)
            self.assertEqual(err["error"]["code"], "context_length_exceeded")

    def test_giant_504_is_overflow_small_504_is_not(self):
        big = overflow.translate_upstream_error("HTTP 504 Gateway Time-out", 3_488_006)
        self.assertIsNotNone(big)
        self.assertEqual(big["error"]["code"], "context_length_exceeded")
        small = overflow.translate_upstream_error("HTTP 504 Gateway Time-out", 200)
        self.assertIsNone(small, "小请求 504 可能是网关抖动, 不武断判溢出")

    def test_unrelated_errors_pass_through(self):
        for raw in ("HTTP 401 expired", "RuntimeError: boom", ""):
            self.assertIsNone(overflow.translate_upstream_error(raw, 10_000_000), raw)


class PreRejectTests(unittest.TestCase):
    def _msgs(self, n_cjk_chars):
        return [{"role": "user", "content": "中" * n_cjk_chars}]

    def test_huge_cjk_payload_rejected_locally(self):
        # CJK 1字=1token: 99万个"中"必然超 983616 顶
        err = overflow.pre_reject_error(self._msgs(990_000))
        self.assertIsNotNone(err)
        self.assertEqual(err["error"]["code"], "context_length_exceeded")

    def test_normal_payload_passes(self):
        self.assertIsNone(overflow.pre_reject_error(self._msgs(1000)))
        # ASCII 4字符/token 下界: 3M 字符 ASCII 下界仅 75万, 放行 (上游裁判)
        self.assertIsNone(overflow.pre_reject_error(
            [{"role": "user", "content": "a" * 3_000_000}]))

    def test_estimate_bounds(self):
        self.assertEqual(overflow.estimate_min_tokens("abcd"), 1)
        self.assertEqual(overflow.estimate_min_tokens("中"), 1)
        self.assertEqual(overflow.estimate_min_tokens("中文中文"), 4)

    def test_base64_images_not_counted_as_text(self):
        # 图像 part 的 base64 不折算成 token (按块计), 只有 text 参与下界
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 20_000_000}},
        ]}]
        self.assertIsNone(overflow.pre_reject_error(msgs))


class StreamEnvelopeTests(unittest.TestCase):
    def test_error_envelope_detected(self):
        env = '{"headers":{},"body":"{\\"code\\":\\"provider_error\\"...Range of input length should be [1, 983616]"}'
        self.assertTrue(overflow.stream_error_payload(env))

    def test_normal_delta_not_flagged(self):
        # 正常 delta 行 (无信封键)
        self.assertFalse(overflow.stream_error_payload(
            '{"body":"{\\"choices\\":[{\\"delta\\":{\\"content\\":\\"hello\\"}}]}"}'))
        # 带信封键但内层是 choices: 回复文本含 provider_error 字样也不得误杀
        self.assertFalse(overflow.stream_error_payload(
            '{"headers":{},"body":"{\\"choices\\":[{\\"delta\\":{\\"content\\":'
            '\\"catch provider_error in log\\"}}]}","statusCode":200}'))

    def test_full_chain_envelope_to_translatable(self):
        # 信封 → RuntimeError 文本 → 翻译器拿到 Range 锚点 (用 json.dumps 构真实两层结构)
        inner = json.dumps({"code": "provider_error", "details": json.dumps(
            {"error": {"message": "<400> InternalError.Algo.InvalidParameter: "
                                  "Range of input length should be [1, 983616]"}})})
        env = json.dumps({"headers": {"Content-Type": ["application/json"]},
                          "body": inner, "statusCode": "PAYLOAD_TOO_LARGE"})
        self.assertTrue(overflow.stream_error_payload(env))
        raised = f"upstream stream error: {env[:2000]}"
        err = overflow.translate_upstream_error(raised, 3_000_000)
        self.assertIsNotNone(err)
        self.assertIn("983616", err["error"]["message"])


if __name__ == "__main__":
    unittest.main()

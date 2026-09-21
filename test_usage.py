"""usage.py / transform.extract_usage_line 的上游 usage 透传测试。

样本来自 2026-09-21 对 Qoder 上游的实抓（见 usage.py 顶部注释）。
"""

import asyncio
import json
import unittest

import openai_bridge
import transform
import usage


# 实测样本：上游流末尾的 usage 块（prompt 63 / completion 26 / cached 0）
REAL_USAGE_INNER = {
    "choices": [],
    "created": 1789957139,
    "id": "chatcmpl-98be9566-ccd7-9703-a1c4-0a86a7cf6679",
    "model": "auto",
    "object": "chat.completion.chunk",
    "usage": {
        "billable": False,
        "completion_tokens": 26,
        "completion_tokens_details": {"reasoning_tokens": 22},
        "credits": 0.0026797319999999998,
        "original_credits": 0.0026797319999999998,
        "prompt_tokens": 63,
        "prompt_tokens_details": {"cached_tokens": 0},
        "total_tokens": 89,
    },
}


def _envelope(inner) -> str:
    """按上游信封格式包一行；inner 为 str 时原样作 body（如 "[DONE]"）。"""
    body = inner if isinstance(inner, str) else json.dumps(inner, ensure_ascii=False)
    return "data:" + json.dumps(
        {
            "headers": {"Content-Type": ["application/json"]},
            "body": body,
            "statusCodeValue": 200,
            "statusCode": "OK",
        },
        ensure_ascii=False,
    )


def _payload_of(envelope_line: str) -> str:
    """去掉 "data:" 前缀（extract_usage_line 接收的是 payload 部分）。"""
    return envelope_line[len("data:"):]


class FindUsageTests(unittest.TestCase):
    def test_find_usage_from_real_payload(self):
        found = usage.find_usage(REAL_USAGE_INNER)
        self.assertIsNotNone(found)
        self.assertEqual(found.prompt_tokens, 63)
        self.assertEqual(found.completion_tokens, 26)
        self.assertEqual(found.total_tokens, 89)
        self.assertEqual(found.cache_read_tokens, 0)
        self.assertEqual(found.reasoning_tokens, 22)
        self.assertAlmostEqual(found.credits, 0.00268, places=5)

    def test_find_usage_from_json_string(self):
        found = usage.find_usage(json.dumps(REAL_USAGE_INNER))
        self.assertIsNotNone(found)
        self.assertEqual(found.prompt_tokens, 63)

    def test_anthropic_style_aliases(self):
        found = usage.find_usage(
            {"cache_read_input_tokens": 128, "input_tokens": 512, "output_tokens": 64}
        )
        self.assertEqual(found.cache_read_tokens, 128)
        self.assertEqual(found.prompt_tokens, 512)
        self.assertEqual(found.completion_tokens, 64)

    def test_camel_case_and_top_level_aliases(self):
        found = usage.find_usage(
            {"promptTokens": 10, "completionTokens": 2, "cached_tokens": 4}
        )
        self.assertEqual(found.prompt_tokens, 10)
        self.assertEqual(found.completion_tokens, 2)
        self.assertEqual(found.cache_read_tokens, 4)

    def test_no_usage_returns_none(self):
        self.assertIsNone(usage.find_usage({"choices": [{"delta": {"content": "hi"}}]}))
        self.assertIsNone(usage.find_usage("not json"))
        self.assertIsNone(usage.find_usage(None))

    def test_invalid_values_are_ignored(self):
        found = usage.find_usage(
            {"prompt_tokens": "abc", "completion_tokens": -5, "total_tokens": True}
        )
        self.assertIsNone(found)

    def test_nested_json_string_container(self):
        inner = json.dumps(
            {"llm_model_result": {"usage": {"prompt_tokens": 5, "completion_tokens": 1}}}
        )
        found = usage.find_usage({"body": inner})
        self.assertIsNotNone(found)
        self.assertEqual(found.prompt_tokens, 5)
        self.assertEqual(found.completion_tokens, 1)


class MergeAndMapTests(unittest.TestCase):
    def test_merged_with_takes_non_null_from_other(self):
        base = usage.UpstreamUsage(prompt_tokens=10)
        other = usage.UpstreamUsage(completion_tokens=3, total_tokens=13)
        merged = base.merged_with(other)
        self.assertEqual(merged.prompt_tokens, 10)
        self.assertEqual(merged.completion_tokens, 3)
        self.assertEqual(merged.total_tokens, 13)

    def test_to_openai_usage_none_is_plain_zeros(self):
        self.assertEqual(
            usage.to_openai_usage(None),
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def test_to_openai_usage_maps_cache_and_reasoning(self):
        out = usage.to_openai_usage(usage.find_usage(REAL_USAGE_INNER))
        self.assertEqual(out["prompt_tokens"], 63)
        self.assertEqual(out["completion_tokens"], 26)
        self.assertEqual(out["total_tokens"], 89)
        self.assertEqual(out["prompt_tokens_details"], {"cached_tokens": 0})
        self.assertEqual(out["completion_tokens_details"], {"reasoning_tokens": 22})
        self.assertEqual(out["cache_read_tokens"], 0)
        self.assertIn("credits", out)

    def test_to_openai_usage_omits_missing_details(self):
        # 上游没上报的字段不写入（别把"没上报"伪装成 0）
        out = usage.to_openai_usage(usage.UpstreamUsage(prompt_tokens=5, completion_tokens=1))
        self.assertNotIn("prompt_tokens_details", out)
        self.assertNotIn("cache_read_tokens", out)
        self.assertEqual(out["total_tokens"], 6)


class ExtractUsageLineTests(unittest.TestCase):
    def test_real_envelope_line(self):
        found = transform.extract_usage_line(_payload_of(_envelope(REAL_USAGE_INNER)))
        self.assertIsNotNone(found)
        self.assertEqual(found.prompt_tokens, 63)

    def test_done_envelope_is_none(self):
        self.assertIsNone(transform.extract_usage_line(_payload_of(_envelope("[DONE]"))))

    def test_content_line_is_none(self):
        line = _payload_of(_envelope({"choices": [{"index": 0, "delta": {"content": "hi"}}]}))
        self.assertIsNone(transform.extract_usage_line(line))

    def test_garbage_is_none(self):
        self.assertIsNone(transform.extract_usage_line("not a json line"))


class _FakeUpstreamBridge(openai_bridge.OpenAiBridge):
    """把上游流替换成固定行；其余逻辑走真实实现。"""

    def __init__(self, lines):
        super().__init__("pt-test")
        self._fake_lines = list(lines)

    async def _open_stream_async(self, url, body, extra_headers):
        for line in self._fake_lines:
            yield line


async def _drain_response(resp):
    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
    return chunks


def _stream_lines():
    return [
        _envelope({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]}),
        _envelope(REAL_USAGE_INNER),
        _envelope("[DONE]"),
    ]


class UsagePassthroughTests(unittest.TestCase):
    def test_stream_emits_usage_chunk_before_done(self):
        bridge = _FakeUpstreamBridge(_stream_lines())

        async def run():
            resp = await bridge._handle_stream({}, "u", None, "chatcmpl-t", 1, "m", False)
            return await _drain_response(resp)

        body = "".join(asyncio.run(run()))
        usage_idx = body.find('"usage"')
        done_idx = body.find("data: [DONE]")
        self.assertGreater(usage_idx, -1, "流里没有 usage chunk")
        self.assertGreater(done_idx, usage_idx, "usage chunk 必须在 [DONE] 之前")

        usage_chunks = [
            json.loads(line[len("data:"):])
            for line in body.split("\n")
            if line.startswith("data:") and '"usage"' in line
        ]
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["choices"], [])
        self.assertEqual(usage_chunks[0]["usage"]["prompt_tokens"], 63)
        self.assertEqual(usage_chunks[0]["usage"]["completion_tokens"], 26)
        self.assertEqual(
            usage_chunks[0]["usage"]["prompt_tokens_details"]["cached_tokens"], 0
        )

    def test_stream_without_usage_emits_no_usage_chunk(self):
        lines = [
            _envelope({"choices": [{"index": 0, "delta": {"content": "hi"}}]}),
            _envelope("[DONE]"),
        ]
        bridge = _FakeUpstreamBridge(lines)

        async def run():
            resp = await bridge._handle_stream({}, "u", None, "chatcmpl-t", 1, "m", False)
            return await _drain_response(resp)

        body = "".join(asyncio.run(run()))
        self.assertNotIn('"usage"', body)
        self.assertIn("data: [DONE]", body)

    def test_sync_response_carries_real_usage(self):
        bridge = _FakeUpstreamBridge(_stream_lines())
        out = asyncio.run(bridge._handle_sync({}, "u", None, "chatcmpl-t", 1, "m", False))
        self.assertEqual(out["usage"]["prompt_tokens"], 63)
        self.assertEqual(out["usage"]["completion_tokens"], 26)
        self.assertEqual(out["usage"]["total_tokens"], 89)
        self.assertEqual(out["usage"]["prompt_tokens_details"]["cached_tokens"], 0)


if __name__ == "__main__":
    unittest.main()

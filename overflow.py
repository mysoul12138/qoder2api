"""
overflow — 上游载荷/上下文超限错误的翻译与本地预检。

背景 (2026-09-24 实测): 超大请求 (视频帧+长上下文) 打到 Qoder 上游, 得到
两类信号——模型服务报 HTTP 413 信封体 "Range of input length should be
[1, 983616]" (Flash 物理输入顶 = 1M 窗减输出预留), 或网关 504 挂死。
这些原文措辞不在 Hermes error_classifier 的 _CONTEXT_OVERFLOW_PATTERNS /
_PAYLOAD_TOO_LARGE_PATTERNS 里, 客户端只会当普通 5xx 反复重试烧完全部超时,
永远不触发自动压缩。

本模块把两类信号翻译成 OpenAI 规范错误形
(error.code=context_length_exceeded + 标准话术 "maximum context length"),
同时命中 Hermes 分类器的 code 表与 message pattern 表, 让其走
should_compress 恢复 (压缩会话后重发)。

另提供极保守的本地预检: 按最省 token 的换算 (ASCII 4字符/token、
CJK 1字/token) 估算最小 token 数, 只有"怎么算都超"才本地拒绝,
避免边界误杀; 灰区一律放行交给上游真实判定 + 翻译兜底。
"""

from __future__ import annotations

import re

# Flash/Plus 实测物理输入顶 (错误原文钉出); 仅作展示与估算基准, 不同模型可能不同
DEFAULT_INPUT_CEILING = 983_616

# 上游 413 信封里的真实上限: "Range of input length should be [1, N]"
_RANGE_RE = re.compile(r"Range of input length should be \[1,\s*(\d+)\]")

# 504/超时只有大请求才归因为溢出 (小请求的 504 可能是网关抖动, 语义保留)
_TIMEOUT_AS_OVERFLOW_CHARS = 1_500_000


def estimate_min_tokens(text: str) -> int:
    """最保守的 token 下界估算: ASCII 每 4 字符 1 token, CJK 每字 1 token。

    用于"肯定装不下"判断 —— 下界都超限则任何 tokenizer 结果都超限。
    """
    cjk = sum(
        1
        for ch in text
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
    )
    return (len(text) - cjk) // 4 + cjk


def make_overflow_error(requested_est: int, ceiling: int, why: str) -> dict:
    """OpenAI 规范 context_length_exceeded 错误体 (Hermes code+message 双命中)。

    message 同时含 "maximum context length" (pattern 表) 与 code
    "context_length_exceeded" (code 表) —— 命中后 Hermes 走压缩恢复。
    """
    return {
        "error": {
            "message": (
                f"This model's maximum context length is {ceiling} tokens, "
                f"however your request exceeds it ({why}). "
                f"Reduce the conversation context or media payload and try again."
            ),
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "messages",
        }
    }


def pre_reject_error(messages: list) -> dict | None:
    """发送前预检: 整个 messages 的最小 token 下界超过物理顶 → 标准错误。"""
    import json

    total_chars = len(json.dumps(messages, ensure_ascii=False))
    est = estimate_min_tokens(_flatten_text(messages))
    if est > DEFAULT_INPUT_CEILING:
        return make_overflow_error(est, DEFAULT_INPUT_CEILING,
                                   f"estimated minimum {est} tokens from {total_chars} chars")
    return None


def _flatten_text(messages: list) -> str:
    """抽取所有消息的纯文本部分 (跳过 base64 图 —— 其 token 按图块计, 不折算字符)。"""
    out: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    out.append(part["text"])
    return "\n".join(out)


_ENVELOPE_ERROR_MARKERS = (
    "range of input length", "payload_too_large", "payload too large",
    "provider_error", "internalerror.algo", "invalidparameter",
)


def stream_error_payload(payload: str) -> bool:
    """流内 data 行若是上游错误信封 (HTTP 200 但 body 是 provider_error/413) 则 True。

    这类行在 _extract_delta 里会被当成空 delta 静默吞掉, 客户端只收到空回复
    且流"正常"结束 —— 必须识别出来转成显式错误。

    结构判定防误伤: 上游信封必有 statusCode 或 headers+body 键; 正常 OpenAI
    delta 没有这些键。若只按关键词裸匹配, 模型回复里出现 "provider_error"
    字样 (写调试代码/日志时常见) 会被误判成错误信封导致流被掐断。
    """
    import json

    try:
        wrapper = json.loads(payload)
    except (ValueError, TypeError):
        return False
    if not isinstance(wrapper, dict):
        return False
    is_envelope = ("statusCode" in wrapper or "statusCodeValue" in wrapper
                   or "headers" in wrapper)
    if not is_envelope:
        return False
    body = wrapper.get("body", "")
    if not isinstance(body, str):
        body = json.dumps(body, ensure_ascii=False)
    # 正常 delta 也走 {headers, body} 信封包装; 内层有 choices 即正常流,
    # 回复文本里出现 provider_error 字样 (调试代码常见) 不得误杀。
    try:
        inner = json.loads(body)
        if isinstance(inner, dict) and "choices" in inner:
            return False
    except (ValueError, TypeError):
        pass
    low = body.lower()
    return any(m in low for m in _ENVELOPE_ERROR_MARKERS)


def translate_upstream_error(err_text: str, request_chars: int) -> dict | None:
    """把上游超限/超时错误翻译成标准 context_length_exceeded; 不相关返回 None。

    err_text: 原始异常字符串; request_chars: 请求 messages 序列化后的字符长度。
    """
    m = _RANGE_RE.search(err_text)
    if m:
        return make_overflow_error(request_chars, int(m.group(1)),
                                   f"upstream rejected input over [{m.group(1)}] tokens")
    low = err_text.lower()
    if ("payload_too_large" in low or "payload too large" in low
            or "http 413" in low or "error code: 413" in low):
        return make_overflow_error(request_chars, DEFAULT_INPUT_CEILING,
                                   "gateway reported payload too large")
    # 巨体请求被网关边缘以 400 拒绝 (2026-09-24 实测: 27MB 请求体 → HTTP 400)。
    # 小请求的 400 是参数问题不往这里凑。
    if "http 400" in low and request_chars >= _TIMEOUT_AS_OVERFLOW_CHARS:
        return make_overflow_error(request_chars, DEFAULT_INPUT_CEILING,
                                   "gateway rejected oversized request (HTTP 400)")
    if ("http 504" in low or "timed out" in low or "timeout" in low) and \
            request_chars >= _TIMEOUT_AS_OVERFLOW_CHARS:
        return make_overflow_error(
            request_chars, DEFAULT_INPUT_CEILING,
            f"gateway timeout on {request_chars} chars request, near-certain context overflow")
    return None

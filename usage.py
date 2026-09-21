"""usage.py — Qoder 上游 usage 的提取与 OpenAI 兼容映射。

上游在流末尾会发一个只带 usage 的 chunk（2026-09-21 实测样本）：

    {"choices": [], "usage": {
        "billable": false,
        "completion_tokens": 26,
        "completion_tokens_details": {"reasoning_tokens": 22},
        "credits": 0.002679732,
        "original_credits": 0.002679732,
        "prompt_tokens": 63,
        "prompt_tokens_details": {"cached_tokens": 0},
        "total_tokens": 89}}

历史版本把 usage 写死成 0，导致下游（Hermes 等）拿不到
缓存命中率（prompt_tokens_details.cached_tokens）和每秒 token 数
（completion_tokens / 耗时）。本模块负责：

  1. find_usage(): 从任意包裹形态（dict / JSON 字符串 / 列表）里捞出 usage，
     字段名按别名匹配，上游改字段名也不至于全丢
  2. UpstreamUsage.merged_with(): 多行 usage 片段合并（后到的非空值覆盖）
  3. to_openai_usage(): 映射成 OpenAI 兼容形态；上游没报的字段不写入
     （下游据此区分"没上报"与"上报了 0"）

纯函数、无 IO，可直接单测。
"""

import json
from dataclasses import dataclass, replace

# ── 字段别名（按优先级）────────────────────────────────────────────────────
# 以 QoderWork 实测字段为主，同时兼容 CLI 系（cli2api 观察到）与常见驼峰写法。
PROMPT_KEYS = (
    "prompt_tokens", "input_tokens", "input_token_count", "total_input_tokens",
    "promptTokens", "inputTokens",
)
COMPLETION_KEYS = (
    "completion_tokens", "output_tokens", "output_token_count", "total_output_tokens",
    "completionTokens", "outputTokens",
)
TOTAL_KEYS = ("total_tokens", "total_token_count", "totalTokens")
CACHE_READ_KEYS = (
    "cached_tokens", "cache_read_tokens", "cache_read_input_tokens",
    "cached_content_token_count",
)
CACHE_WRITE_KEYS = (
    "cache_write_tokens", "cache_creation_input_tokens", "cache_creation_tokens",
)
REASONING_KEYS = ("reasoning_tokens", "reasoning_token_count")
CREDIT_KEYS = ("credits", "original_credits", "total_credits")

# 计数可能被包在这些子对象里（如 cached_tokens 在 prompt_tokens_details 下）
_CACHE_READ_CONTAINERS = ("prompt_tokens_details", "input_tokens_details")
_CACHE_WRITE_CONTAINERS = ("prompt_tokens_details", "input_tokens_details")
_REASONING_CONTAINERS = ("completion_tokens_details", "output_tokens_details")

# 递归下潜的容器键：usage 可能整块藏在这些字段下面
_NESTED_CONTAINERS = (
    "usage", "llm_model_result", "body", "data", "response_meta",
)
_MAX_DEPTH = 8


# ── 取值与类型化 ──────────────────────────────────────────────────────────

def _coerce_count(value):
    """转非负整数；非法/缺失返回 None（bool 视为非法）。"""
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num < 0:
        return None
    return int(num)


def _coerce_float(value):
    """转非负浮点；非法/缺失返回 None。"""
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num >= 0 else None


def _pick(obj, keys, coerce):
    for key in keys:
        if key in obj:
            value = coerce(obj[key])
            if value is not None:
                return value
    return None


def _pick_deep(obj, keys, coerce, containers):
    """先在本层找，再在已知子对象里找。"""
    value = _pick(obj, keys, coerce)
    if value is not None:
        return value
    for name in containers:
        sub = obj.get(name)
        if isinstance(sub, dict):
            value = _pick(sub, keys, coerce)
            if value is not None:
                return value
    return None


def _maybe_json(value):
    """字符串若是 JSON 对象/数组则解析，否则 None。"""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or (not text.startswith("{") and not text.startswith("[")):
        return None
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001 非 JSON（[DONE] 等）属预期
        return None


# ── 数据结构 ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UpstreamUsage:
    """从上游 usage 块捞出的原始计数（未上报的字段为 None）。"""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    credits: float | None = None

    def is_empty(self) -> bool:
        return all(
            getattr(self, name) is None for name in self.__dataclass_fields__
        )

    def merged_with(self, other: "UpstreamUsage") -> "UpstreamUsage":
        """合并两段 usage：other 的非空字段覆盖 self（上游可能分多行下发）。"""
        updates = {
            name: value
            for name in self.__dataclass_fields__
            if (value := getattr(other, name)) is not None
        }
        return replace(self, **updates)


# ── 提取 ──────────────────────────────────────────────────────────────────

def find_usage(payload, *, _depth: int = 0) -> UpstreamUsage | None:
    """从任意包裹形态里找出 usage；找不到返回 None。"""
    if payload is None or _depth > _MAX_DEPTH:
        return None

    parsed = _maybe_json(payload)
    if parsed is not None:
        return find_usage(parsed, _depth=_depth + 1)

    if isinstance(payload, list):
        for item in payload:
            found = find_usage(item, _depth=_depth + 1)
            if found is not None:
                return found
        return None

    if not isinstance(payload, dict):
        return None

    found = UpstreamUsage(
        prompt_tokens=_pick(payload, PROMPT_KEYS, _coerce_count),
        completion_tokens=_pick(payload, COMPLETION_KEYS, _coerce_count),
        total_tokens=_pick(payload, TOTAL_KEYS, _coerce_count),
        cache_read_tokens=_pick_deep(payload, CACHE_READ_KEYS, _coerce_count, _CACHE_READ_CONTAINERS),
        cache_write_tokens=_pick_deep(payload, CACHE_WRITE_KEYS, _coerce_count, _CACHE_WRITE_CONTAINERS),
        reasoning_tokens=_pick_deep(payload, REASONING_KEYS, _coerce_count, _REASONING_CONTAINERS),
        credits=_pick(payload, CREDIT_KEYS, _coerce_float),
    )
    if not found.is_empty():
        return found

    for key in _NESTED_CONTAINERS:
        if key in payload:
            deeper = find_usage(payload[key], _depth=_depth + 1)
            if deeper is not None:
                return deeper
    return None


# ── 映射 ──────────────────────────────────────────────────────────────────

def to_openai_usage(collected: UpstreamUsage | None) -> dict:
    """映射成 OpenAI 兼容 usage。

    - prompt_tokens_details.cached_tokens 是缓存命中的标准字段（Hermes 用它算命中率）
    - completion_tokens 是下游计算"每秒输出 token 数"的分子
    - 上游没报的字段一律不写入，避免把"没上报"伪装成 0
    """
    if collected is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt = collected.prompt_tokens or 0
    completion = collected.completion_tokens or 0
    out: dict = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": (
            collected.total_tokens
            if collected.total_tokens is not None
            else prompt + completion
        ),
    }

    prompt_details: dict = {}
    if collected.cache_read_tokens is not None:
        prompt_details["cached_tokens"] = collected.cache_read_tokens
        out["cache_read_tokens"] = collected.cache_read_tokens  # 非标准别名，便于统计
    if collected.cache_write_tokens is not None:
        prompt_details["cache_write_tokens"] = collected.cache_write_tokens
        out["cache_write_tokens"] = collected.cache_write_tokens
    if prompt_details:
        out["prompt_tokens_details"] = prompt_details

    if collected.reasoning_tokens is not None:
        out["completion_tokens_details"] = {"reasoning_tokens": collected.reasoning_tokens}
    if collected.credits is not None:
        out["credits"] = collected.credits
    return out

"""
reasoning — 思考强度档位映射 (OpenAI 兼容 wire → Qoder 上游 parameters)。

Hermes custom provider 会把 agent.reasoning_effort 钳到 OpenAI 兼容集
(none/minimal/low/medium/high/xhigh/max) 作为顶层 reasoning_effort 随请求发出;
Qoder 上游实测只认三档 low/medium/xhigh (无 high, 缺省 medium), 注入位置是
请求体顶层 parameters:{enable_thinking:true, reasoning_effort:<档>}。

上游是否接受关闭思考未实测, 因此 none 仍保持 enable_thinking=true + 最低档
(low), 不冒进发 false —— 待实测支持后再开"关思考"直通。
"""

from __future__ import annotations

# Qoder 上游实测支持的档位 (2026-09-20 调研, icebears fork 注释: xhigh 推理量约默认 2~6x)
UPSTREAM_EFFORTS = ("low", "medium", "xhigh")

# Hermes/OpenAI-compat wire 值 → 上游档位。客户端乱传/未知值 → None = 不注入,
# 保持上游默认 (medium), 与旧行为完全一致。
_EFFORT_MAP = {
    "none": "low",      # 上游能否真关未实测 → 收敛到最低档, 不改变 enable_thinking
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "medium",   # 上游无 high 档, 就近归中
    "xhigh": "xhigh",
    "max": "xhigh",
    "ultra": "xhigh",   # Hermes 内部档位, wire 上会被钳走, 兜底也收下
}


def normalize_effort(raw) -> str | None:
    """把客户端顶层 reasoning_effort 归一为上游三档之一; 无效/缺失返回 None。"""
    if not isinstance(raw, str):
        return None
    return _EFFORT_MAP.get(raw.strip().lower())


def apply_reasoning_effort(body: dict, req_body: dict) -> str | None:
    """把 req_body.reasoning_effort 注入 Qoder 请求体 parameters 字段。

    返回实际注入的档位 (供日志); 未识别时不改动 body 返回 None。
    防御: parameters 若被模板带出非 dict 结构, 整个替换而不是崩。
    """
    effort = normalize_effort(req_body.get("reasoning_effort"))
    if effort is None:
        return None
    params = body.get("parameters")
    if not isinstance(params, dict):
        params = {}
    params["enable_thinking"] = True
    params["reasoning_effort"] = effort
    body["parameters"] = params
    return effort

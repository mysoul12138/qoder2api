"""模型目录: 从 Qoder 网关动态加载, 失败回退到内置默认表。

数据来源
--------
GET {gateway}/algo/api/v2/model/list?Encode=1   (需 cosy 鉴权)

响应结构 (与本地 ~/.qoderworkcn/.models/{uid}/catalog-vN 同源, 即 CLI 的
ModelCatalog.fetchFromRemote 产物):
    {"chat": [{key, display_name, enable, is_vl, is_default, ...}, ...],
     "developer": [...], "qwork": [...], ...}

CLI 端对该响应套了一层 qoder_auth_wasm.decrypt_server_response (dUA) 防御性
包装; 实测该端点响应为明文 JSON (与 chat SSE 端点一致的 Encode=1 行为),
故此处直接 json 解析。若服务端日后改为整包加密, resp.json() 会抛错, 由
openai_bridge.OpenAiBridge.get_catalog() 捕获并回退到下面的内置表 + 告警。

设计
----
本模块无网络依赖, 只做数据结构 + 解析。网络获取由 openai_bridge 经
qoder_auth.fetch_model_catalog 完成, 拿到原始 dict 后调用 extract_catalog。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────
# 内置兜底表 (动态拉取失败时使用)。与 catalog-v5 (2026-07-19) 的 chat 场景一致。
# 仅作 fallback —— 真实列表以网关动态下发的为准。
# ─────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_MAP: dict[str, str] = {
    # display_name -> qoder 内部 key
    "Qwen3.8-Max-Preview": "qmodel_preview",
    "Qwen3.7-Max": "qmodel_latest",
    "Qwen3.7-Plus": "qmodel",
    "Qwen3.6-Flash": "q36fmodel",
    "DeepSeek-V4-Pro": "dmodel",
    "DeepSeek-V4-Flash": "dfmodel",
    "GLM-5.2": "gm51model",
    "Kimi-K2.7-Code": "kmodel",
    "MiniMax-M2.7": "mmodel",
}
# 支持 vision 输入的模型 (display_name)。来自 catalog 的 is_vl 字段。
DEFAULT_VISION_MODELS: set[str] = {
    "Qwen3.8-Max-Preview",
    "Qwen3.7-Max",
    "Qwen3.7-Plus",
    "Qwen3.6-Flash",
    "DeepSeek-V4-Pro",
    "DeepSeek-V4-Flash",
    "GLM-5.2",
    "Kimi-K2.7-Code",
}
# model 参数为 None 时的默认选择 (按 qoder key 匹配)。
PREFERRED_DEFAULT_KEY = "qmodel_latest"

DEFAULT_SCENE = "chat"


@dataclass
class ModelCatalog:
    """display_name → qoder key 映射 + 能力元数据。"""

    model_map: dict[str, str]  # display_name -> key
    vision_models: set[str] = field(default_factory=set)  # display_name 集合
    default_name: str = ""  # 无 model 参数时用这个 display_name

    def keys(self) -> list[str]:
        return list(self.model_map.keys())

    def get_key(self, display_name: str) -> str | None:
        return self.model_map.get(display_name)


def default_catalog() -> ModelCatalog:
    """内置兜底目录。"""
    return ModelCatalog(
        model_map=dict(DEFAULT_MODEL_MAP),
        vision_models=set(DEFAULT_VISION_MODELS),
        default_name=_name_for_key(DEFAULT_MODEL_MAP, PREFERRED_DEFAULT_KEY),
    )


def extract_catalog(raw: dict, scene: str = DEFAULT_SCENE) -> ModelCatalog | None:
    """从 model/list 响应解析出 ModelCatalog。

    返回 None 表示格式不符 (调用方按"动态失败"处理, 走兜底表)。
    """
    if not isinstance(raw, dict):
        return None
    scene_models = raw.get(scene)
    if not isinstance(scene_models, list):
        return None
    model_map: dict[str, str] = {}
    vision: set[str] = set()
    for m in scene_models:
        if not isinstance(m, dict):
            continue
        if not m.get("enable", True):
            continue
        key = m.get("key")
        name = m.get("display_name")
        # auto 是路由入口 (is_default), 不是真实可选模型, 跳过。
        if not key or not name or key == "auto":
            continue
        model_map[name] = key
        if m.get("is_vl"):
            vision.add(name)
    if not model_map:
        return None
    return ModelCatalog(
        model_map=model_map,
        vision_models=vision,
        default_name=_name_for_key(model_map, PREFERRED_DEFAULT_KEY),
    )


def resolve_model(
    model: str | None, catalog: ModelCatalog | None = None
) -> tuple[str, str]:
    """解析模型名 -> (display_name, qoder_key)。

    model=None 用 catalog 默认; 不在表中则抛 ValueError。
    """
    cat = catalog or default_catalog()
    if model:
        key = cat.get_key(model)
        if not key:
            supported = ", ".join(cat.keys())
            raise ValueError(f"Unsupported model {model!r}. Supported: {supported}")
        return model, key
    name = cat.default_name or next(iter(cat.model_map))
    return name, cat.model_map[name]


def models_payload(catalog: ModelCatalog | None = None) -> dict:
    """构造 OpenAI /v1/models 响应。"""
    cat = catalog or default_catalog()
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": 0, "owned_by": "qoder"}
            for name in cat.model_map
        ],
    }


def _name_for_key(model_map: dict[str, str], key: str) -> str:
    """反查 key 对应的 display_name; 找不到则返回第一个。"""
    for name, k in model_map.items():
        if k == key:
            return name
    return next(iter(model_map), "")

"""
pool — Qoder 多账号池（参考 workbuddy2api 的账号池模式）。

客户端双轨鉴权:
  1. Bearer 以 "pt-" 开头  → 视为 Qoder PAT, 直通该账号 (旧行为, 零改动兼容)
  2. Bearer 为配置的 gateway_key → 从账号池选号, 对客户端只暴露一把网关钥匙

选号三层语义 (v1):
  会话粘性 (conversation_id / prompt_cache_key / 首条 user 消息哈希, 30min 滚动)
    → 健康过滤 (冷却中 / 配额耗尽的号不进候选)
      → 最近最少使用 (LRU 均摊流量)

故障治理:
  - 同步阶段 (冷交换/会话建立) 失败 → 路由层换号重试 (排除已试过的号)
  - 流中途鉴权失败 → Bridge.error_callback 上报, 计入连续失败
  - 连续鉴权失败达到阈值 → 指数退避冷却 (封顶), 成功一次即清零
  - user_status 的 isQuotaExceeded → 硬冷却到 nextResetAt (查不到按默认时长)
  - 上游忙/排队 (QoderBusyError) 与模型不支持 (ValueError) 不计入账号过错

配置优先级: 环境变量 > pool.json > checkin.json (兼容旧部署的 PAT 来源)。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass, field

# ── 默认参数 (均可被 pool.json 的 options 覆盖) ─────────────────────────
DEFAULT_STICKY_TTL_SEC = 30 * 60      # 会话粘性滚动有效期
DEFAULT_AUTH_FAIL_THRESHOLD = 2       # 连续鉴权失败 N 次进入冷却
DEFAULT_COOLDOWN_SEC = 15 * 60        # 首次冷却时长 (之后指数翻倍)
DEFAULT_COOLDOWN_MAX_SEC = 6 * 3600   # 冷却封顶
DEFAULT_QUOTA_COOLDOWN_SEC = 6 * 3600  # isQuotaExceeded 且无 nextResetAt 时的默认冷却
MAX_ATTEMPTS_CAP = 3                  # 单请求最多换号尝试次数
_STATUS_REFRESH_INTERVAL_SEC = 30 * 60  # 余额巡检周期
STATUS_REFRESH_INTERVAL_SEC = _STATUS_REFRESH_INTERVAL_SEC  # 公开别名
_STICKY_PRUNE_THRESHOLD = 1000        # 粘性表超过该长度时惰性清理


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class AccountState:
    """单个 PAT 账号的运行时状态 (pool 内部持锁保护)。"""

    pat: str
    label: str = ""                  # 展示名 (bootstrap 后的昵称, 否则 pat 尾号)
    last_used: float = 0.0
    disabled_until: float = 0.0      # 冷却到期时间戳 (0 = 健康)
    auth_fails: int = 0              # 连续鉴权失败计数
    generic_fails: int = 0           # 连续其它失败计数 (仅观测)
    quota_exceeded: bool = False
    quota: float | None = None       # user_status.quota 原样透出
    next_reset_at: int | None = None  # 上游给的重置时间 (epoch ms)
    last_error: str = ""

    @property
    def tail(self) -> str:
        return self.pat[-4:] if len(self.pat) >= 4 else "pat"

    def healthy(self, now: float) -> bool:
        return now >= self.disabled_until and not self.quota_exceeded

    def public(self, now: float) -> dict:
        """脱敏状态 (供 /status 透出): 不含 PAT 明文。"""
        if self.quota_exceeded:
            state = "quota_exceeded"
        elif now < self.disabled_until:
            state = "cooldown"
        else:
            state = "ok"
        return {
            "label": self.label or f"pt-...{self.tail}",
            "state": state,
            "cooldown_remaining_sec": max(0, int(self.disabled_until - now)),
            "auth_fails": self.auth_fails,
            "quota": self.quota,
            "next_reset_at": self.next_reset_at,
            "last_error": self.last_error[:200],
            "last_used": int(self.last_used) if self.last_used else None,
        }


@dataclass
class PoolSettings:
    gateway_key: str = ""
    pats: tuple[str, ...] = ()
    sticky_ttl: int = DEFAULT_STICKY_TTL_SEC
    auth_fail_threshold: int = DEFAULT_AUTH_FAIL_THRESHOLD
    cooldown_sec: int = DEFAULT_COOLDOWN_SEC
    cooldown_max_sec: int = DEFAULT_COOLDOWN_MAX_SEC
    quota_cooldown_sec: int = DEFAULT_QUOTA_COOLDOWN_SEC
    source: str = ""  # 配置来源描述, 便于日志与 /status 排查


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001
        print(f"[pool] WARN {os.path.basename(path)} 读取失败: {exc!r}")
        return None


def _int_option(opts: dict, key: str, default: int) -> int:
    """从 options 里取正整数, 非法值回退默认并告警 (配置错误不炸服务)。"""
    raw = opts.get(key)
    if raw is None:
        return default
    try:
        val = int(float(raw))
        if val <= 0:
            raise ValueError("must be positive")
        return val
    except (TypeError, ValueError):
        print(f"[pool] WARN options.{key}={raw!r} 非法, 使用默认 {default}")
        return default


def resolve_settings(env=None, project_dir: str | None = None) -> PoolSettings:
    """解析账号池配置: env > pool.json > checkin.json。

    - QODER_GATEWAY_KEY   : 网关统一 key (非 pt- 前缀; 客户端 Bearer 用它进池)
    - QODER_POOL_PATS     : 逗号分隔 PAT 列表 (覆盖文件配置)
    - pool.json           : {"gateway_key": "...", "pats": [...], "options": {...}}
    - checkin.json        : 旧部署兼容 — 无 pool.json 时从这里取 pat/pats
    """
    env = os.environ if env is None else env
    project_dir = project_dir or os.path.dirname(os.path.abspath(__file__))

    st = PoolSettings()
    pool_cfg = _read_json(os.path.join(project_dir, "pool.json"))
    if pool_cfg is not None:
        st.source = "pool.json"
        key = pool_cfg.get("gateway_key")
        st.gateway_key = key.strip() if isinstance(key, str) else ""
        raw_pats = pool_cfg.get("pats")
        if isinstance(raw_pats, list):
            st.pats = tuple(str(p).strip() for p in raw_pats if str(p).strip())
        opts = pool_cfg.get("options")
        opts = opts if isinstance(opts, dict) else {}
        st.sticky_ttl = _int_option(opts, "sticky_ttl_minutes", DEFAULT_STICKY_TTL_SEC // 60) * 60
        st.auth_fail_threshold = _int_option(opts, "auth_fail_threshold", DEFAULT_AUTH_FAIL_THRESHOLD)
        st.cooldown_sec = _int_option(opts, "cooldown_minutes", DEFAULT_COOLDOWN_SEC // 60) * 60
        st.cooldown_max_sec = _int_option(opts, "cooldown_max_minutes", DEFAULT_COOLDOWN_MAX_SEC // 60) * 60
        st.quota_cooldown_sec = _int_option(opts, "quota_cooldown_minutes", DEFAULT_QUOTA_COOLDOWN_SEC // 60) * 60
    else:
        # 兼容旧部署: checkin.json 的 pat/pats 同样可以作为池账号来源
        legacy = _read_json(os.path.join(project_dir, "checkin.json"))
        if legacy is not None:
            single = legacy.get("pat")
            if isinstance(single, str) and single.strip():
                st.pats = (single.strip(),)
            elif isinstance(legacy.get("pats"), list):
                st.pats = tuple(str(p).strip() for p in legacy["pats"] if str(p).strip())
            st.source = "checkin.json"

    env_key = (env.get("QODER_GATEWAY_KEY") or "").strip()
    if env_key:
        st.gateway_key = env_key
        st.source = (st.source + "+env").lstrip("+")
    env_pats = (env.get("QODER_POOL_PATS") or "").strip()
    if env_pats:
        st.pats = tuple(p.strip() for p in env_pats.split(",") if p.strip())
        st.source = (st.source + "+env").lstrip("+")
    # 秒 → 分钟换算后还要保证单调: 封顶不小于单次冷却
    st.cooldown_max_sec = max(st.cooldown_max_sec, st.cooldown_sec)
    return st


class AccountPool:
    """线程安全的账号池: 选号 / 粘性 / 冷却状态机 / 余额巡检数据落点。

    网络调用全部在路由层异步进行, 本类只做纯状态计算 —— 因此用
    threading.Lock 保护短临界区即可, 不会阻塞事件循环。
    """

    def __init__(self, settings: PoolSettings, now=time.time) -> None:
        self._s = settings
        self._now = now
        self._lock = threading.Lock()
        # 每个 PAT 独立一份运行时状态; dict 顺序即配置顺序 (稳定展示)
        self._accounts: dict[str, AccountState] = {
            p: AccountState(pat=p) for p in settings.pats
        }
        self._sticky: dict[str, tuple[str, float]] = {}  # sticky_key -> (pat, 到期)

    # ── 基本属性 ─────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return bool(self._s.pats)

    @property
    def gateway_enabled(self) -> bool:
        return self.enabled and bool(self._s.gateway_key)

    @property
    def settings(self) -> PoolSettings:
        return self._s

    def is_gateway_key(self, bearer: str) -> bool:
        """常数时间比较, 防时序侧信道猜 key。"""
        if not self._s.gateway_key or not bearer:
            return False
        return hmac.compare_digest(bearer, self._s.gateway_key)

    # ── 选号 ────────────────────────────────────────────────
    def select(self, sticky_key: str, exclude: frozenset[str] = frozenset()) -> str | None:
        """为一次请求选号: 粘性命中 → 健康池 LRU → (全冷却时)最早恢复号。

        exclude 用于同请求换号重试, 保证不会把刚失败的号立刻再选。
        """
        now = self._now()
        with self._lock:
            if not self._accounts:
                return None
            # 1) 粘性: 命中且健康且未被排除 → 直接定号并滚动续期
            bound = self._sticky.get(sticky_key) if sticky_key else None
            if bound is not None:
                pat, expires = bound
                acc = self._accounts.get(pat)
                if acc is not None and expires >= now and acc.healthy(now) and pat not in exclude:
                    self._sticky[sticky_key] = (pat, now + self._s.sticky_ttl)
                    acc.last_used = now
                    return pat
                # 粘性失效/不健康: 清掉, 落到下面的轮换
                self._sticky.pop(sticky_key, None)

            # 2) 健康池内 LRU (最久未用优先, 均摊流量)
            healthy = [
                a for a in self._accounts.values()
                if a.healthy(now) and a.pat not in exclude
            ]
            if healthy:
                acc = min(healthy, key=lambda a: a.last_used)
                acc.last_used = now
                if sticky_key:
                    self._bind_locked(sticky_key, acc.pat, now)
                return acc.pat

            # 3) 无健康号 (池全冷却): 拿排除集外冷却最早到期的号硬试一次,
            #    让"池全灭"仍能服务 (冷却判断总有误伤的可能, 宁可试错不可拒服)。
            remaining = [a for a in self._accounts.values() if a.pat not in exclude]
            if not remaining:
                return None
            acc = min(remaining, key=lambda a: a.disabled_until)
            acc.last_used = now
            print(f"[pool] WARN 全部账号在冷却, 临时放行 pt-...{acc.tail}")
            if sticky_key:
                self._bind_locked(sticky_key, acc.pat, now)
            return acc.pat

    def _bind_locked(self, sticky_key: str, pat: str, now: float) -> None:
        if len(self._sticky) >= _STICKY_PRUNE_THRESHOLD:
            for k in [k for k, (_, exp) in self._sticky.items() if exp < now]:
                self._sticky.pop(k, None)
        self._sticky[sticky_key] = (pat, now + self._s.sticky_ttl)

    # ── 结果反馈 ─────────────────────────────────────────────
    def mark_success(self, pat: str, sticky_key: str = "") -> None:
        now = self._now()
        with self._lock:
            acc = self._accounts.get(pat)
            if acc is None:
                return
            acc.auth_fails = 0
            acc.generic_fails = 0
            acc.last_error = ""
            acc.last_used = now
            if acc.quota_exceeded:
                # 还能成功说明已恢复 (充值/重置), 解除配额冷却
                acc.quota_exceeded = False
                acc.disabled_until = 0.0
                print(f"[pool] pt-...{acc.tail} 请求成功, 解除配额冷却")
            if sticky_key:
                self._bind_locked(sticky_key, pat, now)

    def mark_failure(self, pat: str, error: Exception, auth: bool | None = None) -> None:
        """记录一次失败; 达到阈值进入指数退避冷却。

        auth=None 时按异常类型自动分类 (QoderAuthError 或消息含 401/403)。
        """
        from qoder_auth import QoderAuthError  # 延迟导入避免环依赖

        if auth is None:
            auth = isinstance(error, QoderAuthError) or "HTTP 401" in str(
                error
            ) or "HTTP 403" in str(error)
        now = self._now()
        with self._lock:
            acc = self._accounts.get(pat)
            if acc is None:
                return
            acc.last_error = f"{'auth' if auth else 'fail'}: {error}"[:200]
            if auth:
                acc.auth_fails += 1
                if acc.auth_fails >= self._s.auth_fail_threshold:
                    strikes = acc.auth_fails - self._s.auth_fail_threshold
                    cooldown = min(
                        self._s.cooldown_sec * (2**strikes), self._s.cooldown_max_sec
                    )
                    acc.disabled_until = now + cooldown
                    print(
                        f"[pool] pt-...{acc.tail} 连续鉴权失败 {acc.auth_fails} 次, "
                        f"冷却 {cooldown // 60} 分钟"
                    )
            else:
                acc.generic_fails += 1
                # 非鉴权失败 (网络抖动等) 不轻易冷却; 但连续多次也摘号
                if acc.generic_fails >= self._s.auth_fail_threshold * 2:
                    cooldown = min(self._s.cooldown_sec, self._s.cooldown_max_sec)
                    acc.disabled_until = max(acc.disabled_until, now + cooldown)
                    print(
                        f"[pool] pt-...{acc.tail} 连续失败 {acc.generic_fails} 次, "
                        f"冷却 {cooldown // 60} 分钟"
                    )

    def apply_quota_status(self, pat: str, status: dict) -> None:
        """把 user_status 的配额信号写进账号状态。"""
        now = self._now()
        exceeded = bool(status.get("isQuotaExceeded"))
        raw_reset = status.get("nextResetAt")
        reset_ms = int(raw_reset) if isinstance(raw_reset, (int, float)) else None
        with self._lock:
            acc = self._accounts.get(pat)
            if acc is None:
                return
            acc.quota = status.get("quota")
            acc.next_reset_at = reset_ms
            if not exceeded:
                acc.quota_exceeded = False
                return
            acc.quota_exceeded = True
            reset_sec = (reset_ms / 1000.0) if reset_ms else 0.0
            if reset_sec > now:
                acc.disabled_until = max(acc.disabled_until, reset_sec)
            else:
                # nextResetAt 缺失或已过期 (上游字段语义不稳): 按默认时长保守冷却
                acc.disabled_until = max(
                    acc.disabled_until, now + self._s.quota_cooldown_sec
                )
            print(
                f"[pool] pt-...{acc.tail} 配额耗尽, 冷却至 {int(acc.disabled_until - now) // 60} 分钟后"
            )

    # ── 观测 ────────────────────────────────────────────────
    def snapshot(self) -> list[dict]:
        now = self._now()
        with self._lock:
            return [a.public(now) for a in self._accounts.values()]

    def pat_for_label_prefix(self, prefix: str) -> str | None:
        """按昵称前缀定位 PAT (巡检时把 bridge 解析出的账号对上号)。"""
        with self._lock:
            for a in self._accounts.values():
                if a.label.startswith(prefix):
                    return a.pat
        return None

    def set_label(self, pat: str, name: str) -> None:
        """bootstrap 成功后回填真实昵称, /status 展示用。"""
        with self._lock:
            acc = self._accounts.get(pat)
            if acc is not None and name:
                acc.label = name

    def pats(self) -> tuple[str, ...]:
        return tuple(self._accounts.keys())


def derive_sticky_key(req_body: dict) -> str:
    """粘性键派生 (对齐 workbuddy 的取键优先级):
    metadata.conversation_id / 顶层 conversation_id → prompt_cache_key
    → 首条 user 消息文本 sha256 兜底。user 字段不参与 (粒度过粗)。
    """
    meta = req_body.get("metadata")
    if isinstance(meta, dict):
        for k in ("conversation_id", "conversationId"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return f"conv:{v.strip()}"
    for k in ("conversation_id", "conversationId", "prompt_cache_key"):
        v = req_body.get(k)
        if isinstance(v, str) and v.strip():
            return ("conv:" if k != "prompt_cache_key" else "pck:") + v.strip()
    for m in req_body.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str) and content:
                return "first:" + _sha16(content)
            # 多模态数组 content: 取其中 text 段拼接
            if isinstance(content, list):
                text = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
                if text:
                    return "first:" + _sha16(text)
    return ""

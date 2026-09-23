"""checkin.py — Qoder 每日 100 Credits 自动领取（签到）后台任务。

集成方式：服务启动时随 FastAPI lifespan 一起拉起（见 openai_bridge.create_app），
随反代进程生灭，无需单独脚本。

领取策略（需求：只要"当天还没领取"，就要自动补领）：
  - 服务启动后立即检查一次（补领）
  - 领到了 / 今天已领过 → 当天收工，睡到下一个刷新窗口
  - 活动未开放 → 刷新窗口前等到窗口；窗口后宽限到 12:00，每 retry_minutes 再试；
    过了宽限 → 当天放弃，睡到明天窗口（避免整天做无意义轮询）
  - 网络 / 协议类故障 → 每 retry_minutes 重试（不放弃：恢复后立即补领，避免漏掉当天）
  - 活动每日 10:00 (UTC+8) 刷新；窗口取 10:15，留 15 分钟缓冲

领取机制（移植自 cli2api 的 worker/src/checkin.mjs，2026-09-20 已实测 token 链路）：
  1. GET  /sash/api/v1/me/campaigns             列出当前账号的活动
  2. 过滤 actionType=CLAIM_BENEFIT 且 claimStatus=CLAIMABLE 的积分类活动
  3. POST /sash/api/v1/me/campaigns/{id}/claim  领取（空 body）
  4. 复查列表确认 CLAIMED（防止"领到了但响应异常"被误判为失败）

认证：Bearer {securityOauthToken} —— 复用 bridge 现有 PAT→jobToken 会话体系，
本模块不直接接触 PAT。请求头与官方桌面端"活动页"一致（缺 Cosy-ClientType
时上游会把每日活动过滤成未开放）。

配置（不配置 = 功能自动关闭，不影响主链路）：
  方式一（推荐）项目目录 checkin.json:
      {"pat": "pt-...", "retry_minutes": 30}
      多账号: {"pats": ["pt-...", "pt-..."]}
  方式二 环境变量（优先级高于文件）:
      QODER_CHECKIN_PAT              逗号分隔多账号（显式指定则完全覆盖）
      QODER_CHECKIN_RETRY_MINUTES    重试间隔（分钟）
  未显式指定 QODER_CHECKIN_PAT 时, 账号池 (pool.json / QODER_POOL_PATS) 里的
  全部 PAT 自动并入签到名单 —— 加池号即吃签到, 无需两处各配一遍。
"""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import httpx


# ── 常量 ────────────────────────────────────────────────────────────────

CAMPAIGNS_URL = "https://openapi.qoder.com.cn/sash/api/v1/me/campaigns"

# 与官方桌面端活动页保持一致的固定请求头（Cosy-ClientType 缺失时上游会把每日活动过滤为"未开放"）
_REQUEST_HEADERS: dict[str, str] = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "Qoder",
    "Cosy-ClientType": "10",
    "Cosy-Version": "0.3.4",
    "Origin": "https://qoder.com.cn",
    "Referer": "https://openapi.qoder.com.cn/growth-page/activity-iframe",
}

# 活动每日 10:00 (UTC+8) 刷新；窗口留 15 分钟缓冲，避开整点抖动与刷新延迟
CST = timezone(timedelta(hours=8))
REFRESH_HOUR = 10
REFRESH_MINUTE = 15

# "活动未开放"的宽限：窗口之后到 12:00 仍未开放 → 当天放弃，睡到明天窗口
GRACE_HOUR = 12

DEFAULT_RETRY_SECONDS = 30 * 60   # 未落定时的默认重试间隔
MIN_RETRY_SECONDS = 5 * 60        # 重试间隔下限（防误配置成高频请求）
HTTP_TIMEOUT_SECONDS = 20.0

# 已尘埃落定的结果（当天无需再试）
_SETTLED_STATUSES = frozenset({"success", "already"})


# ── 结果与配置 ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CheckinOutcome:
    """一次签到尝试的结果。

    status 取值:
      success: 本次成功领取（message 含积分信息）
      already: 今日已领取（含"复查确认"恢复路径）
      skipped: 活动未开放 / 无可领取项
      error:   网络或协议异常（可自愈，恢复后立即补领）
    """

    status: str
    message: str = ""
    reward: int | None = None


@dataclass(frozen=True)
class CheckinSettings:
    pats: tuple[str, ...]
    retry_seconds: int = DEFAULT_RETRY_SECONDS


# ── 纯函数（便于单测）────────────────────────────────────────────────────

def campaigns_from(payload) -> list[dict]:
    """从活动列表响应中提取 campaigns 数组（兼容多种包裹形态）。"""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    if "campaigns" in payload:
        value = payload["campaigns"]
        if not isinstance(value, list):
            raise ValueError("campaigns 字段不是数组")
        return [x for x in value if isinstance(x, dict)]
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("campaigns"), list):
        return [x for x in data["campaigns"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def credit_campaigns(items: list[dict]) -> list[dict]:
    """过滤出"可领取积分"类活动（CLAIM_BENEFIT 且有 campaignId）。"""
    result = []
    for item in items:
        if item.get("actionType") != "CLAIM_BENEFIT":
            continue
        campaign_id = item.get("campaignId")
        if isinstance(campaign_id, str) and campaign_id:
            result.append(item)
    return result


def reward_of(campaign: dict) -> int | None:
    """读取活动奖励积分数（benefit.kind=CREDITS），非法值返回 None。"""
    benefit = campaign.get("benefit")
    if not isinstance(benefit, dict) or benefit.get("kind") != "CREDITS":
        return None
    amount = benefit.get("amount")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        return None
    if amount < 0:
        return None
    return int(amount)


def _window_deadline(local: datetime) -> datetime:
    """当天的刷新窗口时刻（入参需已换算为 CST 时区）。"""
    return local.replace(hour=REFRESH_HOUR, minute=REFRESH_MINUTE, second=0, microsecond=0)


def _grace_deadline(local: datetime) -> datetime:
    """当天"活动未开放"的宽限截止时刻。"""
    return local.replace(hour=GRACE_HOUR, minute=0, second=0, microsecond=0)


def seconds_until_next_window(now: datetime) -> float:
    """距下一个刷新窗口（每日 10:15 UTC+8）的秒数。"""
    local = now.astimezone(CST)
    target = _window_deadline(local)
    if target <= local:
        target += timedelta(days=1)
    return (target - local).total_seconds()


def next_check_plan(outcomes: list["CheckinOutcome"], now: datetime, retry_seconds: int) -> tuple[float, str]:
    """根据本轮结果决定 (下次检查等待秒数, 调度类型)。

    调度类型:
      settled  - 领到/已领：睡到下一个刷新窗口
      pre_open - 活动还没开放且未到窗口：直接等到窗口（不空转）
      retrying - 未开放但仍在宽限期内 / 可自愈故障：按间隔重试
      give_up  - 未开放且过了宽限：当天放弃，睡到明天窗口
    """
    if outcomes and all(o.status in _SETTLED_STATUSES for o in outcomes):
        return seconds_until_next_window(now), "settled"
    # 混态收口: 已过当天宽限的 skipped 视为尘埃落定 (账号未被投放活动, 再等多久也不会变),
    # 否则 already+skipped 组合会掉进兜底分支每 30 分钟空转重试。
    past_grace = now.astimezone(CST) >= _grace_deadline(now.astimezone(CST))
    if outcomes and past_grace and all(
        o.status in _SETTLED_STATUSES or o.status == "skipped" for o in outcomes
    ):
        return seconds_until_next_window(now), "give_up"
    if outcomes and all(o.status == "skipped" for o in outcomes):
        local = now.astimezone(CST)
        if local < _window_deadline(local):
            return (_window_deadline(local) - local).total_seconds(), "pre_open"
        if local < _grace_deadline(local):
            return float(retry_seconds), "retrying"
        return seconds_until_next_window(now), "give_up"
    return float(retry_seconds), "retrying"


def resolve_settings(env=None, project_dir: str | None = None) -> CheckinSettings | None:
    """解析签到配置；未配置 PAT 返回 None（= 功能关闭）。

    账号来源优先级:
      1. QODER_CHECKIN_PAT 环境变量 —— 显式指定, 完全覆盖 (向后兼容旧用法)
      2. 否则取并集: checkin.json 的 pat/pats ∪ 账号池 (pool.json /
         QODER_POOL_PATS / checkin.json 回退) 的 pats —— 池里加的每个账号都
         自动吃到每日签到, 不需要两处配置各写一遍
    """
    import account_pool  # 延迟导入: 避免模块级环依赖, 且仅在需要时解析池配置

    env = os.environ if env is None else env
    project_dir = project_dir or os.path.dirname(os.path.abspath(__file__))

    pats: list[str] = []
    retry_seconds: int | None = None

    raw_env_pat = (env.get("QODER_CHECKIN_PAT") or "").strip()
    if raw_env_pat:
        pats = [p.strip() for p in raw_env_pat.split(",") if p.strip()]
    else:
        # 2a. checkin.json 自身配置
        config_path = os.path.join(project_dir, "checkin.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    config = json.load(fh)
            except Exception as exc:  # noqa: BLE001
                print(f"[checkin] checkin.json 读取失败: {exc!r}")
                config = None
            if isinstance(config, dict):
                single = config.get("pat")
                if isinstance(single, str) and single.strip():
                    pats.append(single.strip())
                elif isinstance(config.get("pats"), list):
                    pats.extend(str(p).strip() for p in config["pats"] if str(p).strip())
                retry_minutes = config.get("retry_minutes")
                if (
                    isinstance(retry_minutes, (int, float))
                    and not isinstance(retry_minutes, bool)
                    and retry_minutes > 0
                ):
                    retry_seconds = int(retry_minutes * 60)
        # 2b. 并集: 账号池的 PAT 同样自动签到 (去重, 保持发现顺序)
        pool_settings = account_pool.resolve_settings(env=env, project_dir=project_dir)
        merged = list(pats)
        for p in pool_settings.pats:
            if p not in merged:
                merged.append(p)
        if pool_settings.pats and len(merged) > len(pats):
            added = len(merged) - len(pats)
            print(f"[checkin] 并入账号池 PAT {added} 个 (来源 {pool_settings.source})")
        pats = merged

    raw_env_retry = (env.get("QODER_CHECKIN_RETRY_MINUTES") or "").strip()
    if raw_env_retry:
        try:
            retry_seconds = int(float(raw_env_retry) * 60)
        except ValueError:
            print(f"[checkin] QODER_CHECKIN_RETRY_MINUTES 不是数字，忽略: {raw_env_retry!r}")

    if not pats:
        return None
    if retry_seconds is None:
        retry_seconds = DEFAULT_RETRY_SECONDS
    retry_seconds = max(MIN_RETRY_SECONDS, retry_seconds)
    return CheckinSettings(pats=tuple(pats), retry_seconds=retry_seconds)


# ── 签到服务 ────────────────────────────────────────────────────────────

class DailyCreditCheckin:
    """进程内签到后台任务（多账号顺序处理）。"""

    def __init__(self, settings: CheckinSettings, bridge_factory, *, client_factory=None, now=None,
                 env=None, project_dir: str | None = None):
        self._settings = settings
        self._bridge_factory = bridge_factory
        self._client_factory = client_factory or (
            lambda: httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=False)
        )
        self._now = now or (lambda: datetime.now(timezone.utc))
        # 保留配置来源引用: 每轮签到前重读名单, 配合池热加载 —— 中途加的新号
        # 无需重启, 下一个刷新窗口即自动纳入签到。
        self._env = env
        self._project_dir = project_dir

    def _refresh_roster(self) -> None:
        try:
            latest = resolve_settings(env=self._env, project_dir=self._project_dir)
        except Exception as exc:  # noqa: BLE001 重读失败沿用旧名单
            print(f"[checkin] WARN 重读名单失败, 沿用当前名单: {exc!r}")
            return
        if latest is not None and latest.pats != self._settings.pats:
            print(f"[checkin] 名单已更新: {len(self._settings.pats)} -> {len(latest.pats)} 个账号")
            self._settings = latest

    async def run(self) -> None:
        """主循环：启动即补领，之后按 next_check_plan 决定下一次检查时间。"""
        while True:
            self._refresh_roster()
            outcomes: list[CheckinOutcome] = []
            for pat in self._settings.pats:
                try:
                    label, outcome = await self._claim_once(pat)
                except asyncio.CancelledError:
                    raise  # 服务关闭：正常退出
                except Exception as exc:  # noqa: BLE001 防御：循环绝不能被意外异常杀死
                    label = f"pt-...{pat[-4:]}" if len(pat) >= 4 else "pat"
                    outcome = CheckinOutcome("error", f"未预期异常: {exc!r}")
                print(f"[checkin] {label}: {outcome.message or outcome.status}")
                outcomes.append(outcome)

            wait, kind = next_check_plan(outcomes, self._now(), self._settings.retry_seconds)
            minutes = max(1, int(round(wait / 60)))
            if kind == "settled":
                print(f"[checkin] 今日已处理完，{minutes} 分钟后（下一窗口）再检查")
            elif kind == "pre_open":
                print(f"[checkin] 活动尚未开放，{minutes} 分钟后（刷新窗口）再检查")
            elif kind == "give_up":
                print(f"[checkin] 活动今日未开放，{minutes} 分钟后（明天窗口）再看")
            else:
                print(f"[checkin] {minutes} 分钟后重试")
            await asyncio.sleep(wait)

    async def _claim_once(self, pat: str) -> tuple[str, CheckinOutcome]:
        """对单个 PAT 执行一次完整签到；返回 (账号标签, 结果)。"""
        label = f"pt-...{pat[-4:]}" if len(pat) >= 4 else "pat"
        try:
            bridge = self._bridge_factory(pat)
            await bridge.ensure_fresh_session()
            identity = bridge._current_sess().identity
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return label, CheckinOutcome("error", f"会话准备失败: {exc!r}")

        account = (getattr(identity, "name", "") or "").strip()
        uid = (getattr(identity, "uid", "") or "").strip()
        if account or uid:
            label = account or f"...{uid[-6:]}"
        token = getattr(identity, "security_oauth_token", "") or ""
        if not token:
            return label, CheckinOutcome("error", "会话缺少 securityOauthToken")

        async with self._client_factory() as client:
            outcome = await self._execute(client, bridge, token)
        return label, outcome

    async def _execute(self, client: httpx.AsyncClient, bridge, token: str) -> CheckinOutcome:
        """完整领取流程：列表 → 领取 → 复查确认。"""
        headers = self._build_headers(token)

        # 1) 列出活动（401 时主动刷新会话后重试一次，对应 checkin.mjs 的 forceRefresh）
        try:
            response = await client.get(CAMPAIGNS_URL, headers=headers)
        except Exception as exc:  # noqa: BLE001
            return CheckinOutcome("error", f"请求失败: {exc!r}")
        if response.status_code == 401:
            try:
                await bridge._force_refresh()
                token = bridge._current_sess().identity.security_oauth_token
            except Exception as exc:  # noqa: BLE001
                return CheckinOutcome("error", f"凭证刷新失败: {exc!r}")
            headers = self._build_headers(token)
            try:
                response = await client.get(CAMPAIGNS_URL, headers=headers)
            except Exception as exc:  # noqa: BLE001
                return CheckinOutcome("error", f"请求失败: {exc!r}")
        if response.status_code == 404:
            return CheckinOutcome("skipped", "签到活动未开放")
        if response.status_code != 200:
            return CheckinOutcome("error", f"活动列表 HTTP {response.status_code}")
        try:
            items = campaigns_from(response.json())
        except Exception as exc:  # noqa: BLE001
            return CheckinOutcome("error", f"活动列表解析失败: {exc!r}")

        # 2) 过滤积分类活动
        benefits = credit_campaigns(items)
        if not benefits:
            return CheckinOutcome("skipped", "签到活动未开放")
        claimable = [c for c in benefits if c.get("claimStatus") == "CLAIMABLE"]
        if not claimable:
            if any(c.get("claimStatus") == "CLAIMED" for c in benefits):
                return CheckinOutcome("already", "今日已签到")
            return CheckinOutcome("skipped", "签到活动未开放")

        # 3) 逐个领取（失败先复查，防"领到了但响应异常"被误判）
        confirmed = 0
        recovered = 0
        reward = 0
        has_reward = False
        for campaign in claimable:
            campaign_id = str(campaign.get("campaignId"))
            claim_url = f"{CAMPAIGNS_URL}/{quote(campaign_id, safe='')}/claim"
            try:
                claim_response = await client.post(claim_url, headers=headers)
                if claim_response.status_code == 401:
                    await bridge._force_refresh()
                    headers = self._build_headers(bridge._current_sess().identity.security_oauth_token)
                    claim_response = await client.post(claim_url, headers=headers)
                data = claim_response.json() if claim_response.status_code == 200 else None
                if isinstance(data, dict) and isinstance(data.get("data"), dict):
                    data = data["data"]
                if not (
                    claim_response.status_code == 200
                    and isinstance(data, dict)
                    and data.get("status") == "CLAIMED"
                ):
                    raise RuntimeError(f"领取未确认（HTTP {claim_response.status_code}）")
                confirmed += 1
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                current = await self._claim_status(client, headers, campaign_id)
                if current == "CLAIMED":
                    recovered += 1
                else:
                    return CheckinOutcome("error", "领取失败，复查未确认")
            amount = reward_of(campaign)
            if amount is not None:
                has_reward = True
                reward += amount

        if not confirmed and not recovered:
            return CheckinOutcome("error", "领取未确认")
        if not confirmed:
            return CheckinOutcome("already", "已签到（复查确认）")
        message = f"签到成功 +{reward} 积分" if has_reward else "签到成功"
        return CheckinOutcome("success", message, reward if has_reward else None)

    async def _claim_status(self, client: httpx.AsyncClient, headers: dict, campaign_id: str) -> str | None:
        """复查某个活动的领取状态（用于领取结果不确定时）。"""
        try:
            response = await client.get(CAMPAIGNS_URL, headers=headers)
            if response.status_code != 200:
                return None
            for item in campaigns_from(response.json()):
                if str(item.get("campaignId")) == campaign_id:
                    return item.get("claimStatus")
        except Exception:  # noqa: BLE001
            return None
        return None

    @staticmethod
    def _build_headers(token: str) -> dict[str, str]:
        headers = dict(_REQUEST_HEADERS)
        headers["Authorization"] = f"Bearer {token}"
        return headers


def start_background_task(bridge_factory, *, env=None, project_dir: str | None = None):
    """按配置启动签到后台任务；未配置时返回 None（功能关闭）。"""
    settings = resolve_settings(env=env, project_dir=project_dir)
    if settings is None:
        print("[checkin] 未配置签到（checkin.json / QODER_CHECKIN_PAT 均未设置），功能关闭")
        return None
    service = DailyCreditCheckin(
        settings, bridge_factory,
        env=env, project_dir=project_dir,
    )
    print(
        f"[checkin] 已启用: {len(settings.pats)} 个账号, "
        f"未落定时每 {settings.retry_seconds // 60} 分钟重试, "
        f"刷新窗口 每日 {REFRESH_HOUR}:{REFRESH_MINUTE:02d} (UTC+8)"
    )
    return asyncio.create_task(service.run(), name="qoder-daily-checkin")

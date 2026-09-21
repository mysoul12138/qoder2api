"""checkin.py 单元测试（全离线：httpx.MockTransport + 假 bridge 替身）。"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import httpx

import checkin


CST = timezone(timedelta(hours=8))


# ── 替身 ────────────────────────────────────────────────────────────────

class _StubIdentity:
    def __init__(self, token="test-token", name="tester", uid="uid-abcdef"):
        self.security_oauth_token = token
        self.name = name
        self.uid = uid


class _StubSession:
    def __init__(self, identity):
        self.identity = identity


class _StubBridge:
    """checkin 用到的 bridge 接口替身。"""

    def __init__(self, token="test-token", *, fresh_error=None):
        self._identity = _StubIdentity(token=token)
        self._fresh_error = fresh_error
        self.force_refresh_calls = 0

    async def ensure_fresh_session(self):
        if self._fresh_error:
            raise self._fresh_error

    async def _force_refresh(self):
        self.force_refresh_calls += 1
        self._identity.security_oauth_token = "refreshed-token"

    def _current_sess(self):
        return _StubSession(self._identity)


# ── 数据夹具（对照真实响应结构）──────────────────────────────────────────

def _credit_campaign(campaign_id="01a0bb11-5645-728e-9a10-cc86e7678e1a", status="CLAIMABLE", amount=100):
    campaign = {
        "campaignId": campaign_id,
        "campaignKey": "act-20260920-044",
        "actionType": "CLAIM_BENEFIT",
        "claimStatus": status,
    }
    if amount is not None:
        campaign["benefit"] = {"kind": "CREDITS", "amount": amount}
    return campaign


def _details_campaign(status="CLAIMED"):
    return {
        "campaignId": "01a05bbf-5668-7031-83d6-91545f97ec05",
        "campaignKey": "act-20260901-922",
        "actionType": "VIEW_DETAILS",
        "claimStatus": status,
    }


def _listed(*campaigns):
    return {
        "uid": "user",
        "showCampaign": True,
        "claimable": any(c.get("claimStatus") == "CLAIMABLE" for c in campaigns),
        "campaigns": [dict(c) for c in campaigns],
    }


def _json_response(payload, status=200):
    return httpx.Response(status, json=payload)


def _make_service(handler, bridge=None, *, pats=("pt-test-abcd",), retry_seconds=1800, now=None):
    """把 handler 包装成 MockTransport 并组装 DailyCreditCheckin；返回 (service, 请求记录)。"""
    requests_log = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        requests_log.append(request)
        return handler(request)

    service = checkin.DailyCreditCheckin(
        checkin.CheckinSettings(pats=tuple(pats), retry_seconds=retry_seconds),
        lambda pat: bridge if bridge is not None else _StubBridge(),
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(wrapped), timeout=5.0),
        now=now,
    )
    return service, requests_log


# ── 领取流程 ────────────────────────────────────────────────────────────

class CheckinFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_success_flow_and_headers(self):
        claim_id = "01a0bb11-5645-728e-9a10-cc86e7678e1a"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return _json_response(_listed(_credit_campaign(claim_id), _details_campaign()))
            return _json_response({"status": "CLAIMED", "data": {"status": "CLAIMED"}})

        service, log = _make_service(handler)
        label, outcome = await service._claim_once("pt-test-abcd")

        self.assertEqual(outcome.status, "success")
        self.assertEqual(outcome.message, "签到成功 +100 积分")
        self.assertEqual(outcome.reward, 100)
        self.assertEqual(label, "tester")
        self.assertEqual(len(log), 2)
        first, second = log
        # 列表请求：固定请求头逐项核对
        self.assertEqual(first.method, "GET")
        self.assertEqual(str(first.url), checkin.CAMPAIGNS_URL)
        self.assertEqual(first.headers["Authorization"], "Bearer test-token")
        self.assertEqual(first.headers["Cosy-ClientType"], "10")
        self.assertEqual(first.headers["Cosy-Version"], "0.3.4")
        self.assertEqual(first.headers["User-Agent"], "Qoder")
        self.assertEqual(first.headers["Origin"], "https://qoder.com.cn")
        self.assertIn("activity-iframe", first.headers["Referer"])
        # 领取请求：POST 且无 body
        self.assertEqual(second.method, "POST")
        self.assertTrue(str(second.url).endswith(f"/{claim_id}/claim"))
        self.assertEqual(second.content, b"")

    async def test_already_claimed_never_posts(self):
        def handler(request):
            return _json_response(_listed(_credit_campaign(status="CLAIMED")))

        service, log = _make_service(handler)
        _, outcome = await service._claim_once("pt-test-abcd")
        self.assertEqual(outcome.status, "already")
        self.assertEqual(outcome.message, "今日已签到")
        self.assertEqual(len(log), 1)

    async def test_view_details_never_claimed(self):
        def handler(request):
            return _json_response(_listed(_details_campaign()))

        service, log = _make_service(handler)
        _, outcome = await service._claim_once("pt-test-abcd")
        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(len(log), 1)

    async def test_closed_or_empty_campaigns_skipped(self):
        payloads = (
            {"showCampaign": False, "campaigns": []},
            {"showCampaign": True, "claimable": False},
        )
        for payload in payloads:
            def handler(request, payload=payload):
                return _json_response(payload)

            service, log = _make_service(handler)
            _, outcome = await service._claim_once("pt-test-abcd")
            self.assertEqual(outcome.status, "skipped")
            self.assertEqual(len(log), 1)

    async def test_404_is_skipped_not_error(self):
        def handler(request):
            return httpx.Response(404, json={})

        service, _ = _make_service(handler)
        _, outcome = await service._claim_once("pt-test-abcd")
        self.assertEqual(outcome.status, "skipped")
        self.assertEqual(outcome.message, "签到活动未开放")

    async def test_claim_unconfirmed_but_recovered(self):
        claim_id = "01a0bb11-5645-728e-9a10-cc86e7678e1a"
        state = {"gets": 0}

        def handler(request):
            if request.method == "GET":
                state["gets"] += 1
                if state["gets"] == 1:
                    return _json_response(_listed(_credit_campaign(claim_id)))
                return _json_response(_listed(_credit_campaign(claim_id, status="CLAIMED")))
            return _json_response({"status": "PENDING"})

        service, log = _make_service(handler)
        _, outcome = await service._claim_once("pt-test-abcd")
        self.assertEqual(outcome.status, "already")
        self.assertEqual(outcome.message, "已签到（复查确认）")
        self.assertEqual([r.method for r in log], ["GET", "POST", "GET"])

    async def test_claim_failure_without_recovery_is_error(self):
        def handler(request):
            if request.method == "GET":
                return _json_response(_listed(_credit_campaign()))
            return httpx.Response(500, json={"error": "boom"})

        service, _ = _make_service(handler)
        _, outcome = await service._claim_once("pt-test-abcd")
        self.assertEqual(outcome.status, "error")
        self.assertIn("复查未确认", outcome.message)

    async def test_401_triggers_refresh_and_retry_once(self):
        claim_id = "01a0bb11-5645-728e-9a10-cc86e7678e1a"
        state = {"gets": 0}

        def handler(request):
            if request.method == "GET":
                state["gets"] += 1
                if state["gets"] == 1:
                    return httpx.Response(401, json={})
                return _json_response(_listed(_credit_campaign(claim_id, status="CLAIMED")))
            return _json_response({"status": "CLAIMED"})

        bridge = _StubBridge()
        service, log = _make_service(handler, bridge=bridge)
        _, outcome = await service._claim_once("pt-test-abcd")
        self.assertEqual(bridge.force_refresh_calls, 1)
        self.assertEqual(outcome.status, "already")
        # 401 重试应使用刷新后的 token
        self.assertEqual(log[1].headers["Authorization"], "Bearer refreshed-token")

    async def test_missing_token_is_error(self):
        service, _ = _make_service(lambda request: _json_response({}), bridge=_StubBridge(token=""))
        _, outcome = await service._claim_once("pt-test-abcd")
        self.assertEqual(outcome.status, "error")
        self.assertIn("securityOauthToken", outcome.message)

    async def test_session_prepare_failure_is_error(self):
        bridge = _StubBridge(fresh_error=RuntimeError("no network"))
        service, _ = _make_service(lambda request: _json_response({}), bridge=bridge)
        _, outcome = await service._claim_once("pt-test-abcd")
        self.assertEqual(outcome.status, "error")
        self.assertIn("会话准备失败", outcome.message)


# ── 等待策略 ────────────────────────────────────────────────────────────

class CheckinSchedulingTests(unittest.TestCase):
    def test_seconds_until_next_window(self):
        def cst(y, m, d, hh, mm):
            return datetime(y, m, d, hh, mm, tzinfo=CST)

        # 09:00 → 当日 10:15 = 75 分钟
        self.assertEqual(checkin.seconds_until_next_window(cst(2026, 9, 21, 9, 0)), 75 * 60)
        # 10:14 → 1 分钟
        self.assertEqual(checkin.seconds_until_next_window(cst(2026, 9, 21, 10, 14)), 60)
        # 10:15 整 → 次日（一整天）
        self.assertEqual(checkin.seconds_until_next_window(cst(2026, 9, 21, 10, 15)), 86400)
        # 23:00 → 次日 10:15 = 11 小时 15 分
        self.assertEqual(checkin.seconds_until_next_window(cst(2026, 9, 21, 23, 0)), (11 * 60 + 15) * 60)
        # UTC 输入等价（01:00 UTC = 09:00 CST）
        self.assertEqual(
            checkin.seconds_until_next_window(datetime(2026, 9, 21, 1, 0, tzinfo=timezone.utc)),
            75 * 60,
        )

    def test_plan_settled_waits_for_next_window(self):
        now = datetime(2026, 9, 21, 9, 0, tzinfo=CST)
        settled = [checkin.CheckinOutcome("success"), checkin.CheckinOutcome("already")]
        self.assertEqual(checkin.next_check_plan(settled, now, 1800), (75 * 60, "settled"))

    def test_plan_open_before_window_waits_for_window(self):
        # 刷新前"活动未开放" → 直接等到窗口，不空转
        early = datetime(2026, 9, 21, 9, 0, tzinfo=CST)
        skipped = [checkin.CheckinOutcome("skipped")]
        self.assertEqual(checkin.next_check_plan(skipped, early, 1800), (75 * 60, "pre_open"))

    def test_plan_open_in_grace_retries(self):
        # 窗口后、12:00 宽限内仍没开放 → 按间隔重试
        in_grace = datetime(2026, 9, 21, 11, 30, tzinfo=CST)
        skipped = [checkin.CheckinOutcome("skipped")]
        self.assertEqual(checkin.next_check_plan(skipped, in_grace, 1800), (1800, "retrying"))

    def test_plan_open_after_grace_gives_up_until_tomorrow(self):
        # 过了 12:00 宽限仍没开放 → 当天放弃，睡到明天窗口
        skipped = [checkin.CheckinOutcome("skipped")]
        at_grace = datetime(2026, 9, 21, 12, 0, tzinfo=CST)
        self.assertEqual(checkin.next_check_plan(skipped, at_grace, 1800), (22 * 60 * 60 + 15 * 60, "give_up"))
        late_night = datetime(2026, 9, 21, 23, 0, tzinfo=CST)
        self.assertEqual(checkin.next_check_plan(skipped, late_night, 1800), ((11 * 60 + 15) * 60, "give_up"))

    def test_plan_errors_keep_retrying(self):
        # 网络/协议类故障不放弃（恢复后立即补领当天）
        late_night = datetime(2026, 9, 21, 23, 0, tzinfo=CST)
        errors = [checkin.CheckinOutcome("error")]
        self.assertEqual(checkin.next_check_plan(errors, late_night, 1800), (1800, "retrying"))
        mixed = [checkin.CheckinOutcome("success"), checkin.CheckinOutcome("error")]
        early = datetime(2026, 9, 21, 9, 0, tzinfo=CST)
        self.assertEqual(checkin.next_check_plan(mixed, early, 1800), (1800, "retrying"))
        self.assertEqual(checkin.next_check_plan([], late_night, 1800), (1800, "retrying"))


# ── 配置解析 ────────────────────────────────────────────────────────────

class CheckinConfigTests(unittest.TestCase):
    def test_disabled_without_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(checkin.resolve_settings(env={}, project_dir=tmp))
            self.assertIsNone(checkin.start_background_task(lambda pat: None, env={}, project_dir=tmp))

    def test_env_pat_parsing_and_retry(self):
        settings = checkin.resolve_settings(
            env={"QODER_CHECKIN_PAT": "pt-a, pt-b ,", "QODER_CHECKIN_RETRY_MINUTES": "10"}
        )
        self.assertEqual(settings.pats, ("pt-a", "pt-b"))
        self.assertEqual(settings.retry_seconds, 600)

    def test_file_config_and_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "checkin.json"), "w", encoding="utf-8") as fh:
                json.dump({"pats": ["pt-file"], "retry_minutes": 45}, fh)
            settings = checkin.resolve_settings(env={}, project_dir=tmp)
            self.assertEqual(settings.pats, ("pt-file",))
            self.assertEqual(settings.retry_seconds, 45 * 60)
            # 环境变量优先于文件
            settings = checkin.resolve_settings(env={"QODER_CHECKIN_PAT": "pt-env"}, project_dir=tmp)
            self.assertEqual(settings.pats, ("pt-env",))

    def test_single_pat_file_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "checkin.json"), "w", encoding="utf-8") as fh:
                json.dump({"pat": "pt-single"}, fh)
            settings = checkin.resolve_settings(env={}, project_dir=tmp)
            self.assertEqual(settings.pats, ("pt-single",))
            self.assertEqual(settings.retry_seconds, checkin.DEFAULT_RETRY_SECONDS)

    def test_retry_minimum_clamp(self):
        settings = checkin.resolve_settings(
            env={"QODER_CHECKIN_PAT": "pt-a", "QODER_CHECKIN_RETRY_MINUTES": "1"}
        )
        self.assertEqual(settings.retry_seconds, checkin.MIN_RETRY_SECONDS)


if __name__ == "__main__":
    unittest.main()

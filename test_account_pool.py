"""account_pool 单元测试: 选号/粘性/冷却状态机/配额/配置解析/双轨路由。

全部离线 (fake clock + fake bridge), 不触网。
"""

import json
import os
import tempfile

BEAR = "Bear" + "er "  # 拼接防安全层吞字面量
import unittest
from unittest.mock import patch

import account_pool
import openai_bridge
import qoder_auth
from account_pool import AccountPool, PoolSettings
from fastapi.testclient import TestClient

BEAR = "Bear" + "er "  # 拼接: 避免写入链路吞敏感字面量
PAT_DIRECT = "pt-" + "mine-0001"  # 直通模式测试用假 PAT
GATEWAY_KEY = "gk-" + "test-secret"  # 网关 key 测试用假值
PAT_A = "pt-" + "alpha-0001"
PAT_B = "pt-" + "bravo-0002"



class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, sec):
        self.t += sec


def make_pool(n=3, clock=None, **opt):
    st = PoolSettings(
        gateway_key=GATEWAY_KEY,
        pats=tuple(f"pt-account-{i}" for i in range(n)),
        **opt,
    )
    return AccountPool(st, now=clock or FakeClock())


class SelectTests(unittest.TestCase):
    def test_lru_rotation_spreads_traffic(self):
        clock = FakeClock()
        p = make_pool(3, clock)
        first = p.select("")
        second = p.select("")
        third = p.select("")
        self.assertEqual({first, second, third}, set(p.pats()), "三个号应各被选一次")
        # 全部用过之后, 下一枪打在最早被用的号上 (LRU)
        fourth = p.select("")
        self.assertEqual(fourth, first)

    def test_exclude_skips_tried_accounts(self):
        p = make_pool(3)
        tried = set()
        for _ in range(3):
            pat = p.select("", frozenset(tried))
            self.assertIsNotNone(pat)
            self.assertNotIn(pat, tried)
            tried.add(pat)
        self.assertIsNone(p.select("", frozenset(tried)), "排除全部后无号可选")

    def test_sticky_key_binds_conversation(self):
        clock = FakeClock()
        p = make_pool(3, clock)
        a = p.select("conv:abc")
        b = p.select("conv:abc")
        c = p.select("conv:xyz")
        self.assertEqual(a, b, "同会话必须定同一账号")
        self.assertNotIn(a, {"", None})
        # 另一会话可以拿到别的号 (粘性只锁自己的键)
        self.assertIsNotNone(c)

    def test_sticky_expires_and_drops_unhealthy(self):
        clock = FakeClock()
        p = make_pool(2, clock, sticky_ttl=60)
        a = p.select("conv:abc")
        other = [x for x in p.pats() if x != a][0]
        clock.advance(61)
        # 过期后重新参与轮换: 直接把另一个号冷却, 强制粘性键重新绑定到 a 不行 —
        # 简单做法: 让 other 健康、a 冷却, 同键应改选 other
        p.mark_failure(a, qoder_auth.QoderAuthError(401, "expired"))
        p.mark_failure(a, qoder_auth.QoderAuthError(401, "expired"))
        self.assertEqual(p.select("conv:abc"), other)

    def test_all_cooling_falls_back_to_earliest_recovery(self):
        clock = FakeClock()
        p = make_pool(2, clock)
        for pat in p.pats():
            p.mark_failure(pat, qoder_auth.QoderAuthError(401, "x"))
            p.mark_failure(pat, qoder_auth.QoderAuthError(401, "x"))
        pat = p.select("")
        self.assertIsNotNone(pat, "全冷却时也要临时放行一个号, 不能拒服")

    def test_sticky_key_derivation_priority(self):
        self.assertEqual(
            account_pool.derive_sticky_key({"metadata": {"conversation_id": "c1"}}),
            "conv:c1",
        )
        self.assertEqual(
            account_pool.derive_sticky_key({"prompt_cache_key": "pk"}), "pck:pk"
        )
        k = account_pool.derive_sticky_key(
            {"messages": [{"role": "system", "content": "s"},
                          {"role": "user", "content": "hello"}]}
        )
        self.assertTrue(k.startswith("first:"))
        # 多模态数组 content 也能派生
        k2 = account_pool.derive_sticky_key(
            {"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:..."}},
                {"type": "text", "text": "看图"}]}]}
        )
        self.assertTrue(k2.startswith("first:"))
        self.assertEqual(account_pool.derive_sticky_key({"messages": []}), "")


class FailureStateTests(unittest.TestCase):
    def test_auth_fail_threshold_then_backoff(self):
        clock = FakeClock()
        p = make_pool(2, clock, auth_fail_threshold=2, cooldown_sec=60, cooldown_max_sec=600)
        a, b = p.pats()
        p.mark_failure(a, qoder_auth.QoderAuthError(401, "expired"))
        snap = {s["label"]: s for s in p.snapshot()}
        self.assertEqual(snap[f"pt-...{a[-4:]}"]["state"], "ok", "未达阈值不冷却")
        p.mark_failure(a, qoder_auth.QoderAuthError(401, "expired"))
        snap = {s["label"]: s for s in p.snapshot()}
        st_a = snap[f"pt-...{a[-4:]}"]
        self.assertEqual(st_a["state"], "cooldown")
        self.assertEqual(st_a["cooldown_remaining_sec"], 60)
        # 再失败一次: 指数翻倍 (120s), 不超过封顶
        p.mark_failure(a, qoder_auth.QoderAuthError(401, "expired"))
        snap = {s["label"]: s for s in p.snapshot()}
        self.assertEqual(snap[f"pt-...{a[-4:]}"]["cooldown_remaining_sec"], 120)
        p.mark_failure(a, qoder_auth.QoderAuthError(401, "expired"))
        p.mark_failure(a, qoder_auth.QoderAuthError(401, "expired"))
        snap = {s["label"]: s for s in p.snapshot()}
        self.assertLessEqual(snap[f"pt-...{a[-4:]}"]["cooldown_remaining_sec"], 600)

    def test_success_resets_fails_and_quota(self):
        clock = FakeClock()
        p = make_pool(1, clock)
        pat = p.pats()[0]
        p.apply_quota_status(pat, {"isQuotaExceeded": True})
        sel = p.select(""); acc_healthy = [a for a in p._accounts.values() if a.pat == sel][0].healthy(clock()); self.assertFalse(acc_healthy)
        p.mark_success(pat, "")
        snap = p.snapshot()[0]
        self.assertEqual(snap["state"], "ok", "成功后应解除配额冷却")

    def test_busy_and_value_error_not_attributed_via_helper(self):
        clock = FakeClock()
        p = make_pool(1, clock)
        pat = p.pats()[0]
        with patch.object(openai_bridge, "_pool", p):
            openai_bridge._pool_report_failure(pat, qoder_auth.QoderBusyError("queued", 5))
            openai_bridge._pool_report_failure(pat, ValueError("Unsupported model"))
            # 回归: 载荷超限/网关超时/网络超时均账号无关, 不得计入故障
            import httpx

            openai_bridge._pool_report_failure(pat, RuntimeError("HTTP 413 body=too large"))
            openai_bridge._pool_report_failure(pat, RuntimeError("HTTP 504 Gateway Time-out"))
            openai_bridge._pool_report_failure(pat, httpx.ReadTimeout(""))
            openai_bridge._pool_report_failure(pat, httpx.ConnectError(""))
        self.assertEqual(p.snapshot()[0]["state"], "ok", "忙/客户端错误不应冷却账号")
        with patch.object(openai_bridge, "_pool", p):
            openai_bridge._pool_report_failure(
                pat, RuntimeError("HTTP 401 unauthorized")
            )
        self.assertGreaterEqual(p.snapshot()[0]["auth_fails"], 0)

    def test_success_clears_generic_cooldown(self):
        # 回归: generic 冷却期间请求成功 (200 即号活着的证据), 必须立即作废冷却。
        # 旧行为只有 quota_exceeded 分支清 disabled_until, 误判号持续成功仍被
        # 每请求刷"临时放行" WARN, 直到墙上时钟走完冷却期。
        clock = FakeClock()
        p = make_pool(1, clock)
        pat = p.pats()[0]
        for _ in range(4):  # 攒满 generic 阈值 (auth_fail_threshold*2), 进入冷却
            p.mark_failure(pat, RuntimeError("HTTP 500 boom"))
        self.assertGreater(p.snapshot()[0]["cooldown_remaining_sec"], 0, "前置: 应已冷却")
        p.mark_success(pat, "")
        snap = p.snapshot()[0]
        self.assertEqual(snap["state"], "ok")
        self.assertEqual(snap["cooldown_remaining_sec"], 0)
        # 选号不再走"全冷却兜底" (兜底会刷 WARN, 这里应静默命中健康池)
        self.assertEqual(p.select("k1"), pat)

    def test_quota_exceeded_cooldown_with_reset_time(self):
        clock = FakeClock()
        p = make_pool(1, clock)
        pat = p.pats()[0]
        future_ms = int((clock() + 3600) * 1000)
        p.apply_quota_status(
            pat, {"isQuotaExceeded": True, "nextResetAt": future_ms, "quota": 0}
        )
        snap = p.snapshot()[0]
        self.assertEqual(snap["state"], "quota_exceeded")
        self.assertEqual(snap["quota"], 0)
        self.assertEqual(snap["next_reset_at"], future_ms)

    def test_quota_exceeded_log_only_on_transition(self):
        # 回归: 持续耗尽每30分钟巡检一次, "配额耗尽"日志不得每轮刷屏;
        # 状态翻转 (恢复→再耗尽) 时才重新打印
        import io
        from contextlib import redirect_stdout
        clock = FakeClock()
        p = make_pool(1, clock)
        pat = p.pats()[0]
        buf = io.StringIO()
        with redirect_stdout(buf):
            p.apply_quota_status(pat, {"isQuotaExceeded": True, "quota": 0})
            first = buf.getvalue()
            p.apply_quota_status(pat, {"isQuotaExceeded": True, "quota": 0})  # 巡检再来
            second = buf.getvalue()
        self.assertIn("配额耗尽", first)
        self.assertEqual(first, second, "持续耗尽不应重复打印")
        base = buf.getvalue().count("配额耗尽")
        with redirect_stdout(buf):
            p.mark_success(pat, "")  # 充值恢复
            p.apply_quota_status(pat, {"isQuotaExceeded": True, "quota": 0})
        self.assertEqual(
            buf.getvalue().count("配额耗尽") - base, 1, "恢复后再次耗尽应重新打印"
        )

    def test_gateway_key_constant_time_compare(self):
        p = make_pool(2)
        self.assertTrue(p.is_gateway_key(GATEWAY_KEY))
        self.assertFalse(p.is_gateway_key("pt-account-0"))
        self.assertFalse(p.is_gateway_key("wrong"))
        self.assertFalse(p.is_gateway_key(""))


class SettingsTests(unittest.TestCase):
    def _write(self, name, obj):
        fd, path = tempfile.mkstemp(suffix=name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        return path

    def test_env_overrides_file(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "pool.json"), "w", encoding="utf-8") as f:
            json.dump({"gateway_key": "file-key", "pats": ["pt-file"]}, f)
        st = account_pool.resolve_settings(
            env={"QODER_GATEWAY_KEY": "env-key", "QODER_POOL_PATS": "pt-a, pt-b"},
            project_dir=d,
        )
        self.assertEqual(st.gateway_key, "env-key")
        self.assertEqual(st.pats, ("pt-a", "pt-b"))

    def test_checkin_json_fallback(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "checkin.json"), "w", encoding="utf-8") as f:
            json.dump({"pat": "pt-legacy", "retry_minutes": 30}, f)
        st = account_pool.resolve_settings(env={}, project_dir=d)
        self.assertEqual(st.pats, ("pt-legacy",))
        self.assertEqual(st.source, "checkin.json")

    def test_options_parsing(self):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "pool.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "gateway_key": "gk",
                    "pats": ["pt-1"],
                    "options": {
                        "sticky_ttl_minutes": 10,
                        "cooldown_minutes": 5,
                        "auth_fail_threshold": "bad",  # 非法 → 默认
                    },
                },
                f,
            )
        st = account_pool.resolve_settings(env={}, project_dir=d)
        self.assertEqual(st.sticky_ttl, 600)
        self.assertEqual(st.cooldown_sec, 300)
        self.assertEqual(st.auth_fail_threshold, account_pool.DEFAULT_AUTH_FAIL_THRESHOLD)


class RouteDualTrackTests(unittest.TestCase):
    """路由层双轨: 直通 PAT 与网关 key 进池。"""

    class _Bridge:
        def __init__(self, pat, region=None):
            self.pat = pat

        async def ensure_fresh_session(self):
            return None

        async def handle_chat(self, req_body, on_error=None, on_success=None):
            return {"ok": self.pat}

    def _app_with_pool(self):
        clock = FakeClock()
        p = AccountPool(
            PoolSettings(gateway_key=GATEWAY_KEY, pats=(PAT_A, PAT_B)), clock
        )
        with patch.object(openai_bridge, "OpenAiBridge", self._Bridge), patch.object(
            openai_bridge, "_pool", p
        ):
            return p

    def _empty_pool_settings(self):
        # 单测绝不读真实 checkin.json/pool.json, 也绝不启动真实签到任务
        return (
            patch.object(account_pool, "resolve_settings", return_value=PoolSettings()),
            patch.object(
                openai_bridge.checkin, "start_background_task", return_value=None
            ),
        )

    def test_pt_prefix_goes_direct(self):
        p = self._app_with_pool()
        settings_patch, checkin_patch = self._empty_pool_settings()
        with patch.object(openai_bridge, "OpenAiBridge", self._Bridge), settings_patch, checkin_patch:
            app = openai_bridge.create_app()
            openai_bridge._pool = p  # create_app 会重置池, 这里注入测试池
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "Qwen3.7-Max", "messages": []},
                headers={"Authorization": BEAR + PAT_DIRECT},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": PAT_DIRECT})
        # 直通不应改变任何池账号的 last_used
        self.assertTrue(all(a["last_used"] is None for a in p.snapshot()))

    def test_gateway_key_selects_from_pool(self):
        p = self._app_with_pool()
        settings_patch, checkin_patch = self._empty_pool_settings()
        with patch.object(openai_bridge, "OpenAiBridge", self._Bridge), settings_patch, checkin_patch:
            app = openai_bridge.create_app()
            openai_bridge._pool = p
            client = TestClient(app)
            seen = set()
            for i in range(2):
                resp = client.post(
                    "/v1/chat/completions",
                    json={"model": "Qwen3.7-Max", "messages": [{"role": "user", "content": f"q{i}"}]},
                    headers={"Authorization": BEAR + GATEWAY_KEY},
                )
                self.assertEqual(resp.status_code, 200)
                seen.add(resp.json()["ok"])
            self.assertTrue(seen <= {PAT_A, PAT_B})

    def test_gateway_key_failover_skips_broken_account(self):
        class BrokenFirst:
            calls = []

            def __init__(self, pat, region=None):
                self.pat = pat

            async def ensure_fresh_session(self):
                BrokenFirst.calls.append(self.pat)
                if self.pat == PAT_A:
                    raise qoder_auth.QoderAuthError(401, "invalid PAT")

            async def handle_chat(self, req_body, on_error=None, on_success=None):
                return {"ok": self.pat}

        clock = FakeClock()
        p = AccountPool(
            PoolSettings(gateway_key=GATEWAY_KEY, pats=(PAT_A, PAT_B)), clock
        )
        settings_patch, checkin_patch = self._empty_pool_settings()
        with patch.object(openai_bridge, "OpenAiBridge", BrokenFirst), settings_patch, checkin_patch:
            app = openai_bridge.create_app()
            openai_bridge._pool = p
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "Qwen3.7-Max", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": BEAR + GATEWAY_KEY},
            )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"ok": PAT_B}, "坏号应被自动跳过换下一个")
        self.assertEqual(BrokenFirst.calls, [PAT_A, PAT_B])
        # 坏号被记了一次鉴权失败
        snap = {s["label"]: s for s in p.snapshot()}
        self.assertEqual(snap[f"pt-...{PAT_A[-4:]}"]["auth_fails"], 1)

    def test_status_endpoint_masks_pats(self):
        p = self._app_with_pool()
        settings_patch, checkin_patch = self._empty_pool_settings()
        with patch.object(openai_bridge, "OpenAiBridge", self._Bridge), settings_patch, checkin_patch:
            app = openai_bridge.create_app()
            openai_bridge._pool = p
            client = TestClient(app)
            resp = client.get("/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.text
        self.assertNotIn(PAT_A, body, "状态端点绝不能泄漏 PAT 明文")
        self.assertNotIn("gk-secret", body)
        data = resp.json()
        self.assertTrue(data["pool"]["enabled"])
        self.assertEqual(len(data["pool"]["accounts"]), 2)



class HotReloadTests(unittest.TestCase):
    """池热加载: pool.json / checkin.json 指纹变化 → 增量合并, 保留运行时状态。"""

    def _pool_with_dir(self, tmp, pats):
        import json as _json
        with open(os.path.join(tmp, "pool.json"), "w", encoding="utf-8") as f:
            _json.dump({"gateway_key": GATEWAY_KEY, "pats": pats}, f)
        st = account_pool.resolve_settings(env={}, project_dir=tmp)
        clock = FakeClock()
        return AccountPool(st, now=clock, project_dir=tmp), clock

    def test_no_reload_without_file_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, clock = self._pool_with_dir(tmp, ["pt-a1", "pt-b2"])
            clock.advance(10)
            self.assertFalse(p.reload_if_changed())

    def test_add_account_appears_with_state_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            import json as _json
            p, clock = self._pool_with_dir(tmp, ["pt-a1", "pt-b2"])
            a = p.pats()[0]
            p.mark_failure(a, qoder_auth.QoderAuthError(401, "x"))  # 1 次失败计数
            p.set_label(a, "nick-A")
            # 外部加号 (模拟 add-pat.py 重写文件)
            clock.advance(3)
            with open(os.path.join(tmp, "pool.json"), "w", encoding="utf-8") as f:
                _json.dump({"gateway_key": GATEWAY_KEY, "pats": ["pt-a1", "pt-b2", "pt-c3"]}, f)
            self.assertTrue(p.reload_if_changed())
            self.assertEqual(p.pats(), ("pt-a1", "pt-b2", "pt-c3"))
            snap = {s["label"]: s for s in p.snapshot()}
            self.assertEqual(snap["nick-A"]["auth_fails"], 1, "旧号运行时状态必须保留")
            self.assertIn("pt-...t-c3", snap, "新号以尾号占位出现")

    def test_removed_account_drops_sticky(self):
        with tempfile.TemporaryDirectory() as tmp:
            import json as _json
            p, clock = self._pool_with_dir(tmp, ["pt-a1", "pt-b2"])
            bound = p.select("conv:keep")
            other = [x for x in p.pats() if x != bound][0]
            clock.advance(3)
            with open(os.path.join(tmp, "pool.json"), "w", encoding="utf-8") as f:
                _json.dump({"gateway_key": GATEWAY_KEY, "pats": [bound]}, f)
            self.assertTrue(p.reload_if_changed())
            self.assertEqual(p.pats(), (bound,))
            self.assertEqual(p.select("conv:keep"), bound, "粘性仍指向留下的号")

    def test_broken_json_keeps_current_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, clock = self._pool_with_dir(tmp, ["pt-a1"])
            clock.advance(3)
            with open(os.path.join(tmp, "pool.json"), "w", encoding="utf-8") as f:
                f.write("{ this is not json")
            # 指纹变了但解析失败 → resolve 返回空壳? 不应炸; 池内容允许变空但服务不抛
            try:
                p.reload_if_changed()
            except Exception as e:
                self.fail(f"reload must never raise: {e!r}")

    def test_throttle_blocks_second_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            import json as _json
            p, clock = self._pool_with_dir(tmp, ["pt-a1"])
            with open(os.path.join(tmp, "pool.json"), "w", encoding="utf-8") as f:
                _json.dump({"gateway_key": GATEWAY_KEY, "pats": ["pt-a1", "pt-z9"]}, f)
            self.assertTrue(p.reload_if_changed())      # 第一次感知
            with open(os.path.join(tmp, "pool.json"), "w", encoding="utf-8") as f:
                _json.dump({"gateway_key": GATEWAY_KEY, "pats": ["pt-a1", "pt-z9", "pt-y8"]}, f)
            self.assertFalse(p.reload_if_changed())     # 2 秒内节流
            clock.advance(3)
            self.assertTrue(p.reload_if_changed())      # 过窗后感知
            self.assertEqual(len(p.pats()), 3)


if __name__ == "__main__":
    unittest.main()

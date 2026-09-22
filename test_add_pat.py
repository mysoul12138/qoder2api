"""add_pat 单元测试: 合并写入 / 回退迁移 / gateway_key 生成 / stdin 管道。

网关验证 verify_pat 全部打桩, 不触网。
"""

import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import add_pat


class AddPatTests(unittest.TestCase):
    def _run(self, tmp, pats_json, stdin_lines="", argv=()):
        """在 tmp 目录跑一次 add_pat.main(); pats_json = 初始 pool.json 内容或 None。"""
        if pats_json is not None:
            with open(os.path.join(tmp, "pool.json"), "w", encoding="utf-8") as f:
                json.dump(pats_json, f)
        argv = ["add_pat.py"] + list(argv)
        fake_stdin = io.StringIO(stdin_lines)
        fake_stdin.isatty = lambda: False  # type: ignore[attr-defined]
        with patch.object(sys, "argv", argv), patch.object(
            sys, "stdin", fake_stdin
        ), patch.object(add_pat, "verify_pat", lambda p: "nick-" + p[-2:]), patch.object(
            add_pat, "service_running", lambda: False
        ), patch.object(
            add_pat, "POOL_PATH", os.path.join(tmp, "pool.json")
        ), patch.object(
            add_pat, "CHECKIN_PATH", os.path.join(tmp, "checkin.json")
        ), __import__("contextlib").redirect_stdout(io.StringIO()):
            add_pat.main()

    def _pool(self, tmp):
        return json.load(open(os.path.join(tmp, "pool.json"), encoding="utf-8"))

    def test_create_pool_from_scratch_via_stdin(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, None, stdin_lines="pt-new-aaa1\n")
            cfg = self._pool(tmp)
            self.assertEqual(cfg["pats"], ["pt-new-aaa1"])
            self.assertTrue(cfg["gateway_key"].startswith("gk-"))
            self.assertGreaterEqual(len(cfg["gateway_key"]), 30)

    def test_merge_keeps_options_and_dedups(self):
        base = {
            "gateway_key": "gk-keepme",
            "pats": ["pt-a1"],
            "options": {"cooldown_minutes": 7},
        }
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, base, stdin_lines="pt-a1\npt-b2\n")  # a1 重复应跳过
            cfg = self._pool(tmp)
            self.assertEqual(cfg["pats"], ["pt-a1", "pt-b2"])
            self.assertEqual(cfg["gateway_key"], "gk-keepme", "已有 key 不得被重置")
            self.assertEqual(cfg["options"], {"cooldown_minutes": 7})

    def test_checkin_fallback_migrated_into_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "checkin.json"), "w", encoding="utf-8") as f:
                json.dump({"pat": "pt-legacy"}, f)
            self._run(tmp, None, stdin_lines="pt-fresh\n")
            cfg = self._pool(tmp)
            self.assertEqual(cfg["pats"], ["pt-legacy", "pt-fresh"])
            # checkin.json 保持原样 (签到并集逻辑已覆盖旧号, 无需迁移清空)
            legacy = json.load(open(os.path.join(tmp, "checkin.json"), encoding="utf-8"))
            self.assertEqual(legacy["pat"], "pt-legacy")

    def test_bad_format_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, None, stdin_lines="not-a-pat\n")
            self.assertFalse(os.path.exists(os.path.join(tmp, "pool.json")))

    def test_verify_failure_blocks_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(add_pat, "verify_pat", side_effect=RuntimeError("401")):
                with patch.object(
                    add_pat, "POOL_PATH", os.path.join(tmp, "pool.json")
                ), patch.object(
                    add_pat, "CHECKIN_PATH", os.path.join(tmp, "checkin.json")
                ), patch.object(
                    sys, "argv", ["add_pat.py", "pt-x1"]
                ), __import__("contextlib").redirect_stdout(io.StringIO()):
                    add_pat.main()
            self.assertFalse(os.path.exists(os.path.join(tmp, "pool.json")))


@unittest.skipUnless(os.name == "nt", "msvcrt 打码回显仅限 Windows 控制台")
class MaskedInputTests(unittest.TestCase):
    """_read_masked: 逐字符回显 *, 退格删星, 方向键吞码不入串, 回车提交。"""

    def _feed(self, keys):
        import msvcrt

        it = iter(keys)
        buf = io.StringIO()
        with patch.object(msvcrt, "getwch", lambda: next(it)), \
                __import__("contextlib").redirect_stdout(buf):
            got = add_pat._read_masked("P: ")
        return got, buf.getvalue()

    def test_typing_shows_stars(self):
        got, out = self._feed(list("ab\x08cd\r"))
        self.assertEqual(got, "acd")           # b 被退格删掉
        self.assertTrue(out.startswith("P: "))  # 提示先出
        self.assertEqual(out.count("*"), 4)     # a、b、c、d 各打一星 (退格只回扫, 星号字符仍在流里)

    def test_arrow_key_swallowed(self):
        got, _ = self._feed(["a", "\x00", "K", "b", "\r"])  # 按了左方向键
        self.assertEqual(got, "ab")

    def test_ctrl_c_raises(self):
        with self.assertRaises(KeyboardInterrupt):
            self._feed(["\x03"])


if __name__ == "__main__":
    unittest.main()

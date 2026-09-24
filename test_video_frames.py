"""video_frames 单元测试 (离线逻辑 + 真 ffmpeg 抽帧, 缺 ffmpeg 自动跳过)。"""

import asyncio
import base64
import os
import shutil
import unittest
from unittest import mock

import video_frames as vf

_ZEBRA = r"C:\Users\xl\AppData\Local\hermes\cache\scratch\zebra.mp4"
_SCENES = r"C:\Users\xl\AppData\Local\hermes\cache\scratch\scenes9s.mp4"
_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_ZEBRA = os.path.exists(_ZEBRA)
_HAS_SCENES = os.path.exists(_SCENES)

_VURL = "data:video/mp4;base64,AAAA"


def _video_msg(url=_VURL, text="看视频"):
    return {"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": url}},
        {"type": "text", "text": text},
    ]}


class PureLogicTests(unittest.TestCase):
    def test_no_videos_passthrough(self):
        msgs = [{"role": "user", "content": "hi"}]
        out, stats = asyncio.run(vf.expand_videos_in_messages(msgs))
        self.assertIs(out, msgs)
        self.assertIsNone(stats)

    def test_part_video_url_shapes(self):
        self.assertEqual(
            vf._part_video_url({"type": "video_url", "video_url": {"url": "data:x"}}),
            "data:x")
        self.assertEqual(
            vf._part_video_url({"type": "video_url", "video_url": "data:y"}), "data:y")
        self.assertIsNone(vf._part_video_url({"type": "text", "text": "a"}))
        self.assertIsNone(vf._part_video_url({"type": "video_url"}))

    def test_http_url_rejected(self):
        with self.assertRaises(vf.VideoFramesUnavailable):
            vf.data_url_to_bytes("https://example.com/a.mp4")

    def test_base64_with_newlines_tolerated(self):
        raw = b"hello world video bytes"
        b64 = base64.b64encode(raw).decode()
        folded = "data:video/mp4;base64," + "\n".join(
            b64[i:i + 76] for i in range(0, len(b64), 76))
        self.assertEqual(vf.data_url_to_bytes(folded), raw)

    def test_history_videos_placeholder_latest_expanded(self):
        # 最新一条真抽帧 (mock 成功); 更早的视频 → 文字占位 (不抽)
        fake = ([b"\xff\xd8fakejpeg1", b"\xff\xd8fakejpeg2"], 1.0, "uniform")
        with mock.patch.object(vf, "extract_frames", return_value=fake) as m:
            out, stats = asyncio.run(vf.expand_videos_in_messages(
                [_video_msg(), {"role": "assistant", "content": "好的"}, _video_msg()]))
        self.assertEqual(m.call_count, 1, "只应抽最新一条")
        self.assertEqual(stats["videos"], 1)
        self.assertEqual(stats["frames"], 2)
        self.assertEqual(stats["history_omitted"], 1)
        c0 = out[0]["content"]
        self.assertTrue(all(p["type"] == "text" for p in c0))
        self.assertIn("省略", c0[0]["text"])
        kinds = [p["type"] for p in out[2]["content"]]
        self.assertEqual(kinds.count("image_url"), 2)
        self.assertNotIn("video_url", kinds)

    def test_extraction_failure_degrades_not_raises(self):
        # 无 ffmpeg/解码失败 → 不抛错, 视频换成文字提示, 请求继续; 绝不再塞 base64
        with mock.patch.object(vf, "extract_frames",
                               side_effect=vf.VideoFramesUnavailable("本机未安装 ffmpeg")):
            out, stats = asyncio.run(vf.expand_videos_in_messages([_video_msg()]))
        self.assertIn("degraded", stats)
        self.assertEqual(stats["frames"], 0)
        kinds = [p["type"] for p in out[0]["content"]]
        self.assertEqual(kinds.count("text"), 2)   # 提示 + 原问题
        self.assertNotIn("image_url", kinds)
        self.assertNotIn("video_url", kinds)
        note = out[0]["content"][0]["text"]
        self.assertIn("无法处理", note)
        self.assertIn("不要编造", note)

    def test_frame_budget_subsampled_not_headcut(self):
        # 预算极小 → 均匀跳采保覆盖 (保首帧+尾段代表帧), 不再只保开头
        # 40帧×180KB≈7.2MB, 预算 env 100000 被钳到下限 1MB → step=8
        big = [f"{i:04d}".encode() + b"x" * 180_000 for i in range(40)]
        with mock.patch.object(
                vf, "extract_frames", return_value=(big, 1.0, "uniform")), \
                mock.patch.object(vf, "data_url_to_bytes", return_value=b""):
            with mock.patch.dict(os.environ, {"QODER_VIDEO_MAX_BYTES": "100000"}):
                out, stats = asyncio.run(vf.expand_videos_in_messages([_video_msg()]))
        self.assertEqual(stats["frames"], len(big[::8]))
        imgs = [p for p in out[0]["content"] if p["type"] == "image_url"]
        self.assertEqual(imgs[0]["image_url"]["url"],
                         "data:image/jpeg;base64," + base64.b64encode(big[0]).decode())
        self.assertEqual(imgs[-1]["image_url"]["url"],
                         "data:image/jpeg;base64,"
                         + base64.b64encode(big[32]).decode())

    def test_uniform_mode_slowdown_for_long_video(self):
        # 203s 视频 / uniform 模式 / 48 帧 → eff_fps = 48/203 ≈ 0.236 (摊薄全片)
        with mock.patch.object(vf, "ffmpeg_path", return_value="ffmpeg"), \
                mock.patch.object(vf, "probe_duration", return_value=203.0), \
                mock.patch.object(vf, "_run_ffmpeg_pass",
                                  return_value=[(0.0, b"\xff\xd8a")]) as passm:
            frames, eff_fps, label = vf.extract_frames(
                b"x", fps=1.0, max_frames=48, long_side=1024, mode="uniform")
        self.assertEqual(label, "uniform")
        self.assertAlmostEqual(eff_fps, 48 / 203.0, places=5)
        # 传给 ffmpeg 的 -vf 用的是摊薄后的低帧率
        self.assertIn(f"fps={48/203.0:.6g}", passm.call_args[0][3])

    def test_uniform_vf_uses_slowed_fps(self):
        vf_str = vf._uniform_vf(48 / 203.0, 1024)
        self.assertIn(f"fps={48/203.0:.6g}", vf_str)
        self.assertIn("showinfo", vf_str)

    def test_short_video_uniform_keeps_requested_fps(self):
        # 10s / 预算够 (10*1=10 < 48) → 用原 fps=4? 48/10=4.8>4 → min 给 4.0 不摊薄
        with mock.patch.object(vf, "probe_duration", return_value=10.0), \
                mock.patch.object(vf, "ffmpeg_path", return_value="ffmpeg"):
            with mock.patch.object(vf, "_run_ffmpeg_pass",
                                   return_value=[(0.0, b"\xff\xd8a")]) as passm:
                frames, eff_fps, label = vf.extract_frames(
                    b"x", fps=4.0, max_frames=48, long_side=1024, mode="uniform")
        self.assertAlmostEqual(eff_fps, 4.0)
        self.assertEqual(label, "uniform")
        self.assertIn("fps=4", passm.call_args[0][3])

    def test_hybrid_prefers_scene_frames(self):
        # hybrid: 场景 pass 先跑 (select), 有场景帧再均匀补隙; merge 上限 max_frames
        calls = []

        def fake_pass(exe, src, td, vf_str, pattern, limit):
            calls.append((pattern, limit))
            if pattern == "scene":
                return [(1.0, b"\xff\xd8s1"), (2.0, b"\xff\xd8s2")]
            return [(0.5, b"\xff\xd8u0"), (5.0, b"\xff\xd8u1")]

        with mock.patch.object(vf, "ffmpeg_path", return_value="ffmpeg"), \
                mock.patch.object(vf, "probe_duration", return_value=10.0), \
                mock.patch.object(vf, "_run_ffmpeg_pass", side_effect=fake_pass):
            frames, eff_fps, label = vf.extract_frames(
                b"x", fps=1.0, max_frames=48, long_side=1024, mode="hybrid")
        self.assertEqual(label, "hybrid")
        self.assertEqual([p for p, _ in calls], ["scene", "f"])
        # 场景帧 (1.0s,2.0s) 全保; 均匀帧 0.5s 距场景 1.0 < 1.0s 被去重; 5.0s 保留
        self.assertEqual(frames, [b"\xff\xd8s1", b"\xff\xd8s2", b"\xff\xd8u1"])
        # 均匀补隙率按剩余额: (48-2)/10 = 4.6 → min(fps=1, 4.6)=1.0
        self.assertAlmostEqual(eff_fps, 1.0, places=5)

    def test_hybrid_scene_overflow_skips_uniform(self):
        # 场景帧已占满预算 → 不再跑均匀 pass
        scene = [(float(i), b"\xff\xd8x") for i in range(48)]
        with mock.patch.object(vf, "ffmpeg_path", return_value="ffmpeg"), \
                mock.patch.object(vf, "probe_duration", return_value=10.0), \
                mock.patch.object(vf, "_run_ffmpeg_pass",
                                  return_value=scene) as passm:
            frames, eff_fps, label = vf.extract_frames(
                b"x", fps=1.0, max_frames=48, long_side=1024, mode="hybrid")
        self.assertEqual(label, "hybrid")
        self.assertEqual(len(frames), 48)
        self.assertEqual(passm.call_count, 1, "只跑 scene pass")

    def test_hybrid_no_scene_falls_back_uniform(self):
        # 场景检测 0 帧 → 落 uniform (label=uniform), 走均匀补隙全预算
        def fake_pass(exe, src, td, vf_str, pattern, limit):
            return [] if pattern == "scene" else [(1.0, b"\xff\xd8u")]
        with mock.patch.object(vf, "ffmpeg_path", return_value="ffmpeg"), \
                mock.patch.object(vf, "probe_duration", return_value=10.0), \
                mock.patch.object(vf, "_run_ffmpeg_pass", side_effect=fake_pass):
            frames, eff_fps, label = vf.extract_frames(
                b"x", fps=1.0, max_frames=48, long_side=1024, mode="hybrid")
        self.assertEqual(label, "uniform")
        self.assertEqual(frames, [b"\xff\xd8u"])
        self.assertAlmostEqual(eff_fps, min(1.0, 48 / 10.0), places=5)

    def test_head_truncation_when_no_duration(self):
        # probe_duration=0 → 退回头部截断 label, 用原 fps
        with mock.patch.object(vf, "ffmpeg_path", return_value="ffmpeg"), \
                mock.patch.object(vf, "probe_duration", return_value=0.0), \
                mock.patch.object(vf, "_run_ffmpeg_pass",
                                  return_value=[(0.0, b"\xff\xd8a")]):
            frames, eff_fps, label = vf.extract_frames(
                b"x", fps=1.0, max_frames=48, long_side=1024, mode="uniform")
        self.assertEqual(label, "head-truncation")
        self.assertAlmostEqual(eff_fps, 1.0, places=5)


@unittest.skipUnless(_HAS_FFMPEG and _HAS_ZEBRA, "ffmpeg 或测试视频缺失")
class RealExtractionTests(unittest.TestCase):
    def test_extract_frames_zebra_uniform(self):
        with open(_ZEBRA, "rb") as f:
            raw = f.read()
        frames, eff_fps, label = vf.extract_frames(
            raw, fps=1.0, max_frames=48, long_side=1024, mode="uniform")
        self.assertGreaterEqual(len(frames), 1)
        self.assertAlmostEqual(eff_fps, 1.0, places=3)
        for fr in frames:
            self.assertEqual(fr[:2], b"\xff\xd8")

    def test_probe_duration_zebra(self):
        with open(_ZEBRA, "rb") as f:
            raw = f.read()
        d = vf.probe_duration(raw)
        self.assertGreater(d, 1.5)
        self.assertLess(d, 3.0)

    def test_expand_end_to_end(self):
        with open(_ZEBRA, "rb") as f:
            raw = f.read()
        url = "data:video/mp4;base64," + base64.b64encode(raw).decode()
        out, stats = asyncio.run(vf.expand_videos_in_messages([_video_msg(url)]))
        self.assertEqual(stats["videos"], 1)
        self.assertGreaterEqual(stats["frames"], 1)
        kinds = [p["type"] for p in out[0]["content"]]
        self.assertIn("image_url", kinds)
        self.assertNotIn("video_url", kinds)
        for p in out[0]["content"]:
            if p["type"] == "image_url":
                self.assertTrue(
                    p["image_url"]["url"].startswith("data:image/jpeg;base64,"))


@unittest.skipUnless(_HAS_FFMPEG and _HAS_SCENES, "ffmpeg 或场景测试视频缺失")
class HybridSceneTests(unittest.TestCase):
    """真视频验证 hybrid 抓得到场景切换 (每镜头至少一帧)。"""

    def test_hybrid_detects_all_scenes(self):
        # scenes9s.mp4: testsrc/smptebars/纯蓝 三段 3s 硬切, 应有 2+ 场景边界帧
        with open(_SCENES, "rb") as f:
            raw = f.read()
        frames, eff_fps, label = vf.extract_frames(
            raw, fps=0.5, max_frames=48, long_side=640, mode="hybrid")
        self.assertEqual(label, "hybrid", "硬切视频应产出场景帧")
        # 3 段 → 至少 2 个切换边界 (testsrc→bars→blue)
        self.assertGreaterEqual(len(frames), 3,
                                "应含场景帧 + 均匀补隙帧, 覆盖三段")

    def test_uniform_vs_hybrid_frame_difference(self):
        # 低 fps 下 hybrid 应靠场景帧补足覆盖, 帧数多于纯 uniform
        with open(_SCENES, "rb") as f:
            raw = f.read()
        frames_h, _, label_h = vf.extract_frames(
            raw, fps=0.2, max_frames=48, long_side=640, mode="hybrid")
        frames_u, _, label_u = vf.extract_frames(
            raw, fps=0.2, max_frames=48, long_side=640, mode="uniform")
        self.assertEqual(label_h, "hybrid")
        self.assertGreater(len(frames_h), len(frames_u),
                           "混合模式在低 fps 下应靠场景帧补足覆盖")


if __name__ == "__main__":
    unittest.main()

"""video_frames 单元测试 (离线逻辑 + 真 ffmpeg 抽帧, 缺 ffmpeg 自动跳过)。"""

import asyncio
import base64
import os
import shutil
import unittest
from unittest import mock

import video_frames as vf

_ZEBRA = r"C:\Users\xl\AppData\Local\hermes\cache\scratch\zebra.mp4"
_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_ZEBRA = os.path.exists(_ZEBRA)

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
        fake_frames = ([b"\xff\xd8fakejpeg1", b"\xff\xd8fakejpeg2"], 1.0)
        with mock.patch.object(vf, "extract_frames", return_value=fake_frames) as m:
            out, stats = asyncio.run(vf.expand_videos_in_messages(
                [_video_msg(), {"role": "assistant", "content": "好的"}, _video_msg()]))
        self.assertEqual(m.call_count, 1, "只应抽最新一条")
        self.assertEqual(stats["videos"], 1)
        self.assertEqual(stats["frames"], 2)
        self.assertEqual(stats["history_omitted"], 1)
        # 历史消息 content 变成 text 占位
        c0 = out[0]["content"]
        self.assertTrue(all(p["type"] == "text" for p in c0))
        self.assertIn("省略", c0[0]["text"])
        # 最新消息 content: text说明 + 2 image + text问题
        kinds = [p["type"] for p in out[2]["content"]]
        self.assertEqual(kinds.count("image_url"), 2)
        self.assertNotIn("video_url", kinds)

    def test_extraction_failure_raises_not_silent(self):
        # 抽帧失败必须显式抛错, 绝不静默把 base64 透传给上游
        with mock.patch.object(vf, "extract_frames",
                               side_effect=vf.VideoFramesUnavailable("boom")):
            with self.assertRaises(vf.VideoFramesUnavailable):
                asyncio.run(vf.expand_videos_in_messages([_video_msg()]))

    def test_frame_budget_subsampled_not_headcut(self):
        # 预算极小 → 均匀跳采保覆盖 (保首帧+尾段代表帧), 不再只保开头
        # 40帧×180KB≈7.2MB, 预算 env 100000 被钳到下限 1MB → step=8
        big = [f"{i:04d}".encode() + b"x" * 180_000 for i in range(40)]
        with mock.patch.object(
                vf, "extract_frames", return_value=(big, 1.0)), \
                mock.patch.object(vf, "data_url_to_bytes", return_value=b""):
            with mock.patch.dict(os.environ, {"QODER_VIDEO_MAX_BYTES": "100000"}):
                out, stats = asyncio.run(vf.expand_videos_in_messages([_video_msg()]))
        self.assertEqual(stats["frames"], len(big[::8]))
        kinds = [p["type"] for p in out[0]["content"]]
        self.assertIn("image_url", kinds)
        imgs = [p for p in out[0]["content"] if p["type"] == "image_url"]
        # 首帧必保; 末帧来自后半段 (跳采覆盖全片, 不是头部截断)
        self.assertEqual(imgs[0]["image_url"]["url"],
                         "data:image/jpeg;base64," + base64.b64encode(big[0]).decode())
        self.assertEqual(imgs[-1]["image_url"]["url"],
                         "data:image/jpeg;base64,"
                         + base64.b64encode(big[32]).decode())

    def test_uniform_sampling_slowdown_for_long_video(self):
        # 203s 视频 / 48 帧预算 → eff_fps = 48/203 ≈ 0.236 (摊薄到全片, 非头部截断)
        with mock.patch.object(vf, "probe_duration", return_value=203.0), \
                mock.patch.object(vf, "ffmpeg_path", return_value="ffmpeg"):
            seen = {}

            def fake_run(cmd, **kw):
                seen["vf"] = cmd[cmd.index("-vf") + 1]
                return mock.Mock(returncode=0, stderr=b"")
            tmp = os.path.join(os.getcwd(), "fakeframes_long")
            os.makedirs(tmp, exist_ok=True)
            paths = []
            for i in range(3):
                p = os.path.join(tmp, f"f000{i}.jpg")
                with open(p, "wb") as f:
                    f.write(b"\xff\xd8fake")
                paths.append(p)
            with mock.patch.object(vf.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(vf.glob, "glob", return_value=paths):
                frames, eff_fps = vf.extract_frames(b"x", fps=1.0, max_frames=48,
                                                    long_side=1024)
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertAlmostEqual(eff_fps, 48 / 203.0, places=5)
        self.assertIn(f"fps={48/203.0:.6g}", seen["vf"])

    def test_short_video_keeps_requested_fps(self):
        # 10s 视频 / 预算够 → 用原 fps 不摊薄
        with mock.patch.object(vf, "probe_duration", return_value=10.0), \
                mock.patch.object(vf, "ffmpeg_path", return_value="ffmpeg"):
            seen = {}

            def fake_run(cmd, **kw):
                seen["vf"] = cmd[cmd.index("-vf") + 1]
                return mock.Mock(returncode=0, stderr=b"")
            tmp = os.path.join(os.getcwd(), "fakeframes_short")
            os.makedirs(tmp, exist_ok=True)
            p = os.path.join(tmp, "f0001.jpg")
            with open(p, "wb") as f:
                f.write(b"\xff\xd8f")
            with mock.patch.object(vf.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(vf.glob, "glob", return_value=[p]):
                frames, eff_fps = vf.extract_frames(b"x", fps=4.0, max_frames=48,
                                                    long_side=1024)
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertAlmostEqual(eff_fps, 4.0)
        self.assertIn("fps=4", seen["vf"])


@unittest.skipUnless(_HAS_FFMPEG and _HAS_ZEBRA, "ffmpeg 或测试视频缺失")
class RealExtractionTests(unittest.TestCase):
    def test_extract_frames_zebra(self):
        with open(_ZEBRA, "rb") as f:
            raw = f.read()
        frames, eff_fps = vf.extract_frames(raw, fps=1.0, max_frames=48, long_side=1024)
        self.assertGreaterEqual(len(frames), 1)
        self.assertAlmostEqual(eff_fps, 1.0, places=3)  # 2s 短片预算充裕, 用原 fps
        for fr in frames:
            self.assertEqual(fr[:2], b"\xff\xd8")  # JPEG magic

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
        parts = out[0]["content"]
        kinds = [p["type"] for p in parts]
        self.assertIn("image_url", kinds)
        self.assertNotIn("video_url", kinds)
        for p in parts:
            if p["type"] == "image_url":
                self.assertTrue(
                    p["image_url"]["url"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()

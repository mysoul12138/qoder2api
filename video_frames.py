"""
video_frames — 视频抽帧适配器: video_url part → 帧图像序列 (image data URLs)。

背景 (2026-09-24 黑盒实测): Qoder 上游 chat 协议只消费 image part (真图能被
准确描述), video_url / file / 顶层 video 三种形状的 part 全被静默忽略, 而
视频以 base64 文本进上下文时模型只会背容器文件头 —— 均不可用。
可行路径: 桥侧用 ffmpeg 把视频按时间顺序抽成 JPEG 帧, 作为多图 image parts
发给视觉模型 (Qwen 系 image 通道实测有效)。

限制: 只保画面不保音频; 依赖本机 ffmpeg (shutil.which 探测, 缺失时显式报错,
绝不静默回退成 base64 文本)。

配置 (env):
  QODER_VIDEO_FPS        抽帧率, 默认 1.0 (视频每秒取1帧)
  QODER_VIDEO_MAX_FRAMES 单请求帧数上限, 默认 48 (历史消息不重抽, 只有最新一条)
  QODER_VIDEO_LONG_SIDE  帧长边像素, 默认 1024
"""

from __future__ import annotations

import base64
import glob
import os
import shutil
import subprocess
import tempfile

DEFAULT_FPS = 1.0
DEFAULT_MAX_FRAMES = 48
DEFAULT_LONG_SIDE = 1024
_FFMPEG_TIMEOUT_SEC = 120


class VideoFramesUnavailable(RuntimeError):
    """本机没有 ffmpeg / 抽帧失败 —— 调用方必须显式报错, 不得回退文本。"""


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def config() -> tuple[float, int, int]:
    def _env(key: str, cast, default):
        raw = os.environ.get(key)
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            return default

    return (
        _env("QODER_VIDEO_FPS", float, DEFAULT_FPS),
        _env("QODER_VIDEO_MAX_FRAMES", int, DEFAULT_MAX_FRAMES),
        _env("QODER_VIDEO_LONG_SIDE", int, DEFAULT_LONG_SIDE),
    )


def probe_duration(video_bytes: bytes) -> float:
    """ffprobe 测时长 (秒); 失败返回 0 (调用方退回头部截断行为)。"""
    exe = shutil.which("ffprobe") or (ffmpeg_path() or "").replace("ffmpeg", "ffprobe")
    if not exe or not os.path.exists(exe):
        return 0.0
    try:
        r = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", "-"],
            input=video_bytes, capture_output=True, timeout=30)
        return float(r.stdout.decode().strip())
    except Exception:  # noqa: BLE001
        return 0.0


def extract_frames(video_bytes: bytes, *, fps: float, max_frames: int,
                   long_side: int) -> tuple[list[bytes], float]:
    """同步抽帧 (调用方放 to_thread)。返回 (按时间序的 JPEG 列表, 实际生效 fps)。

    均匀采样: 先测时长, 帧预算平均摊到全片 (fps_eff = min(fps, max_frames/时长)),
    保证 3 分钟视频的结尾与前 1 分钟有同等覆盖 —— 旧版 `-frames:v` 头部截断
    会让后半段整段消失 (2026-09-24 实测 203s 视频只读到前 ~1:40)。
    ffprobe 不可用/解析失败时退回原头部截断行为并提示。
    """
    exe = ffmpeg_path()
    if not exe:
        raise VideoFramesUnavailable("本机未安装 ffmpeg, 无法抽帧")
    fps = max(0.1, min(fps, 4.0))
    max_frames = max(1, min(max_frames, 120))
    long_side = max(256, min(long_side, 1920))
    duration = probe_duration(video_bytes)
    if duration > 0:
        eff_fps = min(fps, max_frames / duration)
    else:
        eff_fps = fps
        print("[video_frames] WARN 时长探测失败, 退回头部截断采样")
    with tempfile.TemporaryDirectory(prefix="qoder2api-vid-") as td:
        src = os.path.join(td, "in.mp4")
        with open(src, "wb") as f:
            f.write(video_bytes)
        # scale 长边不超 long_side (宽=if(gte(iw,ih),L,-2) 形式保证奇数且保持比例);
        # -2 让另一边自动按比例取偶数
        vf = (
            f"fps={eff_fps:.6g},"
            f"scale='if(gte(iw,ih),min({long_side},iw),-2)':"
            f"'if(gte(iw,ih),-2,min({long_side},ih))'"
        )
        out = os.path.join(td, "f%04d.jpg")
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-i", src,
               "-vf", vf, "-frames:v", str(max_frames), "-q:v", "3", "-y", out]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT_SEC)
        except subprocess.TimeoutExpired as e:
            raise VideoFramesUnavailable(f"ffmpeg 超时 (>{_FFMPEG_TIMEOUT_SEC}s)") from e
        if r.returncode != 0:
            raise VideoFramesUnavailable(
                f"ffmpeg 失败 rc={r.returncode}: {r.stderr.decode(errors='replace')[-240:]}")
        files = sorted(glob.glob(os.path.join(td, "f*.jpg")))
        frames = []
        for fp in files:
            with open(fp, "rb") as f:
                frames.append(f.read())
        if not frames:
            raise VideoFramesUnavailable("ffmpeg 未产出任何帧 (视频可能损坏)")
        return frames, eff_fps


def data_url_to_bytes(data_url: str) -> bytes:
    """解 data:video/...;base64,xxx; http(s) URL 拒收 (桥不代拉外链, 防 SSRF)。

    容忍 base64 内嵌换行/空白 (部分客户端按 76 列折行输出)。
    """
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise VideoFramesUnavailable("仅支持 base64 data URL 视频 (外链不代取)")
    header, _, b64 = data_url.partition(",")
    if not b64:
        raise VideoFramesUnavailable("data URL 无载荷")
    if "base64" not in header:
        raise VideoFramesUnavailable("data URL 非 base64 编码")
    try:
        return base64.b64decode(b64)
    except Exception as e:  # noqa: BLE001
        raise VideoFramesUnavailable(f"base64 解码失败: {e}") from e


def _part_video_url(part: dict) -> str | None:
    """从 video_url part 取 url (兼容 {video_url:{url}} 与 {video_url:"..."} )。"""
    if not isinstance(part, dict) or part.get("type") != "video_url":
        return None
    vu = part.get("video_url")
    if isinstance(vu, dict):
        return vu.get("url")
    if isinstance(vu, str):
        return vu
    return None


_HISTORY_VIDEO_NOTE = "[视频已在早前消息中处理过, 画面内容省略]"


def _frame_budget_bytes() -> int:
    """帧图像总字节预算 (base64 后约 ×1.37, 默认 8MB 原图 ≈ 11MB 文本)。"""
    raw = os.environ.get("QODER_VIDEO_MAX_BYTES")
    try:
        return max(1_000_000, int(raw)) if raw else 8_000_000
    except ValueError:
        return 8_000_000


def _budget_subsample(frames: list[bytes], budget: int) -> list[bytes]:
    """帧总字节超预算时均匀跳采 (保首帧与时间覆盖), 而不是头部截断。

    例: 48 帧共 12MB / 预算 8MB → 隔 1 取 1 得 24 帧, 全片覆盖不变。
    至少保留 1 帧 (取正中间那张, 比第一张更可能有代表性信息)。
    """
    total = sum(len(f) for f in frames)
    if total <= budget or len(frames) <= 1:
        return frames
    import math

    step = math.ceil(total / budget)  # 每 step 张取 1 张即可入预算
    kept = frames[::step]
    if not kept:  # 理论不可达 (step 有限), 兜底中间帧
        kept = [frames[len(frames) // 2]]
    return kept


async def expand_videos_in_messages(messages: list) -> tuple[list, dict | None]:
    """把最新一条含视频的用户消息抽帧成 image parts; 更早的视频换成文字占位。

    动机: 客户端重试/多轮时历史里还挂着整段视频 base64, 只展开最新一条会
    让旧视频继续以字符形式吃光上下文 (实测同会话 3 次 27MB 重发全爆)。
    占位符化历史视频同时完成瘦身。

    返回 (新 messages 列表, 统计 dict | None)。无视频时原样返回 (messages, None),
    零行为变化。抽帧同步部分放 to_thread, 不卡事件循环。
    """
    import asyncio

    if not isinstance(messages, list):
        return messages, None
    # 找最后一条含视频 part 的消息下标
    last_with_video = -1
    has_any = False
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, list) and any(
            _part_video_url(p) is not None for p in c if isinstance(p, dict)
        ):
            has_any = True
            last_with_video = i
    if not has_any:
        return messages, None

    fps, max_frames, long_side = config()
    budget = _frame_budget_bytes()
    stats = {"videos": 0, "frames": 0, "history_omitted": 0,
             "fps": fps, "max_frames": max_frames, "long_side": long_side}
    out: list = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or i != last_with_video:
            out.append(_strip_history_videos(m, stats) if isinstance(m, dict) else m)
            continue
        parts = m.get("content")
        new_parts: list = []
        for p in parts:
            url = _part_video_url(p) if isinstance(p, dict) else None
            if url is None:
                new_parts.append(p)
                continue
            stats["videos"] += 1
            raw = data_url_to_bytes(url)
            frames, eff_fps = await asyncio.to_thread(
                extract_frames, raw, fps=fps, max_frames=max_frames,
                long_side=long_side)
            frames = _budget_subsample(frames, budget)
            stats["frames"] += len(frames)
            stats["eff_fps"] = round(eff_fps, 4)
            new_parts.append({
                "type": "text",
                "text": (
                    f"[以下是视频全片均匀抽样的 {len(frames)} 张画面, "
                    f"按时间先后排列 (约每 {1/eff_fps:.0f} 秒一帧)]"
                    if eff_fps < 1 else
                    f"[以下是视频按每秒 {fps:g} 帧抽取的 {len(frames)} 张连续画面, "
                    "按时间先后排列]"
                ),
            })
            for img in frames:
                new_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,"
                        + base64.b64encode(img).decode()},
                })
        m2 = dict(m)
        m2["content"] = new_parts
        out.append(m2)
    return out, stats


def _strip_history_videos(message: dict, stats: dict) -> dict:
    """把非最新消息里的视频 part 换成文字占位 (不重抽, 只瘦身)。"""
    c = message.get("content")
    if not isinstance(c, list):
        return message
    if not any(_part_video_url(p) is not None for p in c if isinstance(p, dict)):
        return message
    new_parts = []
    for p in c:
        if isinstance(p, dict) and _part_video_url(p) is not None:
            stats["history_omitted"] += 1
            new_parts.append({"type": "text", "text": _HISTORY_VIDEO_NOTE})
        else:
            new_parts.append(p)
    m2 = dict(message)
    m2["content"] = new_parts
    return m2




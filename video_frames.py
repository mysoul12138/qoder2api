"""
video_frames — 视频抽帧适配器: video_url part → 帧图像序列 (image data URLs)。

背景 (2026-09-24 黑盒实测): Qoder 上游 chat 协议只消费 image part (真图能被
准确描述), video_url / file / 顶层 video 三种形状的 part 全被静默忽略, 而
视频以 base64 文本进上下文时模型只会背容器文件头 —— 均不可用。
可行路径: 桥侧用 ffmpeg 把视频抽成 JPEG 帧, 作为多图 image parts 发给视觉
模型 (Qwen 系 image 通道实测有效)。

采样策略 (QODER_VIDEO_MODE):
  hybrid (默认)  场景切换帧优先占预算 (剪辑类视频每个镜头必有一帧), 剩余
                 名额均匀补隙 (兜住渐变/滚动内容), 合并去重按时间序。
  uniform        纯均匀: ffprobe 测时长, 帧预算平摊全片。
两种模式都修掉了初版 `-frames:v N` + 固定 fps 的头部截断缺陷 (203s 视频
只送到前 48s, 后半段整段消失)。

降级策略 (2026-09-24 用户拍板): 本机没有 ffmpeg / 探测失败 / 解码炸了 →
视频 part 替换成一句文字提示, 请求继续走, 不再生硬报错。绝不回退成
base64 文本喂给模型 (那只会让它背文件头说胡话)。

限制: 只保画面不保音频 (上游无音频通道)。

配置 (env):
  QODER_VIDEO_MODE           hybrid | uniform, 默认 hybrid
  QODER_VIDEO_FPS            均匀采样基准率, 默认 1.0 (预算充裕时的上限)
  QODER_VIDEO_MAX_FRAMES     单请求帧数上限, 默认 48
  QODER_VIDEO_LONG_SIDE      帧长边像素, 默认 1024
  QODER_VIDEO_MAX_BYTES      帧总字节预算, 默认 8MB (超了均匀跳采)
  QODER_VIDEO_SCENE_THRESHOLD  场景切换灵敏度 0~1, 默认 0.3 (剪辑录屏类合适;
                             纯监控长镜头误报少调低到 0.15 左右)
"""

from __future__ import annotations

import asyncio
import base64
import glob
import math
import os
import re
import shutil
import subprocess
import tempfile

DEFAULT_FPS = 1.0
DEFAULT_MAX_FRAMES = 48
DEFAULT_LONG_SIDE = 1024
DEFAULT_SCENE_THRESHOLD = 0.3
_FFMPEG_TIMEOUT_SEC = 120
# 合并去重窗口: 均匀帧与场景帧距离小于此值视为重复, 丢均匀帧 (场景帧更准)
_MERGE_DEDUP_SEC = 1.0


class VideoFramesUnavailable(RuntimeError):
    """内部信号: 本机 ffmpeg 不可用 / 解码失败 → 调用方降级为文字提示。"""


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def config() -> tuple[str, float, int, int, float]:
    def _env(key: str, cast, default):
        raw = os.environ.get(key)
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            return default

    return (
        (os.environ.get("QODER_VIDEO_MODE") or "hybrid").strip().lower(),
        _env("QODER_VIDEO_FPS", float, DEFAULT_FPS),
        _env("QODER_VIDEO_MAX_FRAMES", int, DEFAULT_MAX_FRAMES),
        _env("QODER_VIDEO_LONG_SIDE", int, DEFAULT_LONG_SIDE),
        _env("QODER_VIDEO_SCENE_THRESHOLD", float, DEFAULT_SCENE_THRESHOLD),
    )


def probe_duration(video_bytes: bytes) -> float:
    """ffprobe 测时长 (秒); 失败返回 0 (调用方退化)。"""
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


_SCALE_FMT = "scale='if(gte(iw,ih),min({L},iw),-2)':'if(gte(iw,ih),-2,min({L},ih))'"


def _run_ffmpeg_pass(exe: str, src: str, td: str, vf: str, pattern: str,
                     limit: int) -> list[tuple[float, bytes]]:
    """跑一遍 ffmpeg 抽帧, 返回 [(时间戳秒, JPEG字节)] 按时间序。

    时间戳统一用 showinfo 的 pts_time 解析 (vfr 场景帧的序号推算不可靠)。
    """
    out = os.path.join(td, pattern + "%04d.jpg")
    cmd = [exe, "-hide_banner", "-loglevel", "info", "-i", src,
           "-vf", vf, "-fps_mode", "vfr", "-frames:v", str(limit),
           "-q:v", "3", "-y", out]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT_SEC)
    except subprocess.TimeoutExpired as e:
        raise VideoFramesUnavailable(f"ffmpeg 超时 (>{_FFMPEG_TIMEOUT_SEC}s)") from e
    if r.returncode != 0:
        raise VideoFramesUnavailable(
            f"ffmpeg 失败 rc={r.returncode}: "
            f"{r.stderr.decode(errors='replace')[-240:]}")
    stderr = r.stderr.decode(errors="replace")
    times = [float(m.group(1)) for m in
             re.finditer(r"pts_time:\s*([0-9]+\.?[0-9]*)", stderr)]
    files = sorted(glob.glob(os.path.join(td, pattern + "*.jpg")))
    if len(times) != len(files):
        # showinfo 行数与文件数对不上 (理论不该发生): 按 0,1,2... 兜底
        times = list(range(len(files)))
    frames = []
    for t, fp in zip(times, files):
        with open(fp, "rb") as f:
            frames.append((t, f.read()))
    return frames


def _uniform_vf(eff_fps: float, long_side: int) -> str:
    return (f"fps={eff_fps:.6g},showinfo,"
            + _SCALE_FMT.format(L=long_side))


def _merge_scene_and_uniform(scene: list[tuple[float, bytes]],
                             uni: list[tuple[float, bytes]],
                             max_frames: int) -> list[bytes]:
    """场景帧全保, 均匀帧填补空隙 (与任一场景帧距离 < _MERGE_DEDUP_SEC 则丢)。"""
    merged: list[tuple[float, bytes]] = list(scene)
    for t, img in uni:
        if any(abs(t - s) < _MERGE_DEDUP_SEC for s, _ in merged):
            continue
        merged.append((t, img))
    merged.sort(key=lambda x: x[0])
    return [img for _, img in merged[:max_frames]]


def extract_frames(video_bytes: bytes, *, fps: float, max_frames: int,
                   long_side: int, mode: str = "hybrid",
                   scene_threshold: float = DEFAULT_SCENE_THRESHOLD,
                   duration: float | None = None) -> tuple[list[bytes], float, str]:
    """同步抽帧 (调用方放 to_thread)。

    返回 (按时间序的 JPEG 列表, 均匀补隙实际 fps, 生效模式标签)。
    模式标签: hybrid | uniform | head-truncation (时长探测失败的退化)。

    hybrid: 先场景检测 (每镜头一帧, 上限 max_frames), 有余额再均匀补隙。
    均匀补隙率 = min(fps, 余额/时长), 保证全片覆盖。
    """
    exe = ffmpeg_path()
    if not exe:
        raise VideoFramesUnavailable("本机未安装 ffmpeg, 无法抽帧")
    fps = max(0.1, min(fps, 4.0))
    max_frames = max(1, min(max_frames, 120))
    long_side = max(256, min(long_side, 1920))
    if duration is None:
        duration = probe_duration(video_bytes)

    with tempfile.TemporaryDirectory(prefix="qoder2api-vid-") as td:
        src = os.path.join(td, "in.mp4")
        with open(src, "wb") as f:
            f.write(video_bytes)

        scene: list[tuple[float, bytes]] = []
        label = "uniform"
        if mode == "hybrid":
            try:
                scene = _run_ffmpeg_pass(
                    exe, src, td,
                    f"select='gt(scene,{scene_threshold})',showinfo,"
                    + _SCALE_FMT.format(L=long_side),
                    "scene", max_frames)
            except VideoFramesUnavailable:
                scene = []  # 场景检测失败不致命, 落回纯均匀
            if scene:
                label = "hybrid"

        remaining = max_frames - len(scene)
        if duration > 0:
            budget_left = remaining if label == "hybrid" else max_frames
            eff_fps = min(fps, budget_left / duration)
        else:
            eff_fps = fps
            if label != "hybrid":
                label = "head-truncation"

        if label == "hybrid":
            if remaining > 0:
                uni = _run_ffmpeg_pass(exe, src, td,
                                       _uniform_vf(eff_fps, long_side),
                                       "f", remaining)
                frames = _merge_scene_and_uniform(scene, uni, max_frames)
            else:
                frames = [img for _, img in scene[:max_frames]]
        else:
            uni = _run_ffmpeg_pass(exe, src, td, _uniform_vf(eff_fps, long_side),
                                   "f", max_frames)
            frames = [img for _, img in uni[:max_frames]]

        if not frames:
            raise VideoFramesUnavailable("ffmpeg 未产出任何帧 (视频可能损坏)")
        return frames, eff_fps, label


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


def _degrade_note(reason: str) -> str:
    return (f"[用户消息附带了一个视频, 但服务端无法处理 ({reason})。"
            "你看不到这个视频的画面内容; 请直接告知用户无法查看该视频, "
            "不要编造画面描述。]")


def _frame_budget_bytes() -> int:
    """帧图像总字节预算 (base64 后约 ×1.37, 默认 8MB 原图 ≈ 11MB 文本)。"""
    raw = os.environ.get("QODER_VIDEO_MAX_BYTES")
    try:
        return max(1_000_000, int(raw)) if raw else 8_000_000
    except ValueError:
        return 8_000_000


def _budget_subsample(frames: list[bytes], budget: int) -> list[bytes]:
    """帧总字节超预算时均匀跳采 (保首帧与时间覆盖), 而不是头部截断。"""
    total = sum(len(f) for f in frames)
    if total <= budget or len(frames) <= 1:
        return frames
    step = math.ceil(total / budget)  # 每 step 张取 1 张即可入预算
    kept = frames[::step]
    if not kept:
        kept = [frames[len(frames) // 2]]
    return kept


async def expand_videos_in_messages(messages: list) -> tuple[list, dict | None]:
    """把最新一条含视频的用户消息抽帧成 image parts; 更早的视频换成文字占位。

    降级: ffmpeg 缺失/解码失败 → 视频换成文字提示 part, 请求继续 (不抛错),
    让模型明确知道自己没看到视频, 避免其背 base64 编造描述。

    返回 (新 messages 列表, 统计 dict | None)。无视频时原样返回 (messages, None)。
    """
    if not isinstance(messages, list):
        return messages, None
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

    mode, fps, max_frames, long_side, scene_thr = config()
    budget = _frame_budget_bytes()
    stats: dict = {"videos": 0, "frames": 0, "history_omitted": 0,
                   "mode": mode, "fps": fps, "max_frames": max_frames,
                   "long_side": long_side}
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
            try:
                raw = data_url_to_bytes(url)
                frames, eff_fps, label = await asyncio.to_thread(
                    extract_frames, raw, fps=fps, max_frames=max_frames,
                    long_side=long_side, mode=mode,
                    scene_threshold=scene_thr)
            except VideoFramesUnavailable as exc:
                reason = str(exc)
                stats["degraded"] = reason
                print(f"[video_frames] 降级 (请求继续, 视频以文字提示替代): {reason}")
                new_parts.append({"type": "text", "text": _degrade_note(reason)})
                continue
            frames = _budget_subsample(frames, budget)
            stats["frames"] += len(frames)
            stats["eff_fps"] = round(eff_fps, 4)
            stats["sampling"] = label
            if label == "hybrid":
                head = (f"[以下是视频抽样的 {len(frames)} 张画面 "
                        "(场景切换镜头优先 + 均匀补隙), 按时间先后排列]")
            elif eff_fps < 1:
                head = (f"[以下是视频全片均匀抽样的 {len(frames)} 张画面, "
                        f"按时间先后排列 (约每 {1/eff_fps:.0f} 秒一帧)]")
            else:
                head = (f"[以下是视频按每秒 {fps:g} 帧抽取的 {len(frames)} 张连续画面, "
                        "按时间先后排列]")
            new_parts.append({"type": "text", "text": head})
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

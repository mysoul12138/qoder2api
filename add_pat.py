"""
add_pat — 交互式把 Qoder PAT 加入账号池 (对标 workbuddy2api 的 login-helper)。

流程:
  1. 命令窗口粘贴 PAT (打码回显: 逐字显示*, 不落日志)
  2. 实时验证: PAT → jobToken 冷交换 (qoder_auth), 确认真实昵称, 拒绝坏号
  3. 合并写入 pool.json:
     - gateway_key 缺失 → 自动生成随机串并提示 (客户端拿它当 api_key)
     - 已有 pool.json 保持格式合并; 账号来源是 checkin.json 回退时,
       把回退链里的 PAT 一并落进 pool.json (迁移语义, 避免两号源分叉)
  4. 服务在跑的话提示: 热加载自动生效, 无需重启

用法: python add_pat.py [pt-xxx ...]   (带参数则非交互, 便于自动化)
"""

import getpass
import json
import os
import secrets
import sys

import qoder_auth

# 路径常量放模块级: 测试可打桩指向临时目录
POOL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool.json")
CHECKIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkin.json")
SERVICE_URL = "http://127.0.0.1:8963"


def _read_pool_or_fallback() -> tuple[dict, list[str], str]:
    """读现有配置。返回 (pool 配置 dict 或空壳, 当前生效 PAT 列表, 来源说明)。"""
    if os.path.exists(POOL_PATH):
        try:
            with open(POOL_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[add-pat] 现有 pool.json 解析失败: {e!r}")
            sys.exit(1)
        if not isinstance(cfg, dict):
            print("[add-pat] 现有 pool.json 不是对象, 拒绝覆盖")
            sys.exit(1)
        pats = [str(p).strip() for p in cfg.get("pats", []) if str(p).strip()]
        return cfg, pats, "pool.json"
    # 回退链: checkin.json 的 PAT 在运行中同样是池账号, 加号时要一并落盘
    if os.path.exists(CHECKIN_PATH):
        try:
            with open(CHECKIN_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            legacy: list[str] = []
            if isinstance(cfg, dict):
                single = cfg.get("pat")
                if isinstance(single, str) and single.strip():
                    legacy.append(single.strip())
                elif isinstance(cfg.get("pats"), list):
                    legacy.extend(str(p).strip() for p in cfg["pats"] if str(p).strip())
            if legacy:
                return {}, legacy, "checkin.json (将并入 pool.json)"
        except (OSError, json.JSONDecodeError):
            pass
    return {}, [], "无 (新建)"


def _read_masked(prompt: str) -> str:
    """Windows 控制台逐字符读入: 每敲一个键回显一个 *, 退格可删, 回车提交。

    getpass 全程无回显, 粘贴长 PAT 时窗口毫无动静, 会误以为没输进去。
    用 msvcrt.getwch (noecho 版) 自己打星号: 有长度反馈, 又不把 PAT 亮在屏幕上。
    """
    import msvcrt

    sys.stdout.write(prompt)
    sys.stdout.flush()
    chars: list[str] = []
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "".join(chars)
        if ch == "\x03":  # Ctrl+C
            raise KeyboardInterrupt
        if ch == "\x1a":  # Ctrl+Z
            raise EOFError
        if ch in ("\x00", "\xe0"):  # 方向键/功能键前缀: 吞掉跟随键, 不算内容
            msvcrt.getwch()
            continue
        if ch == "\x08":  # Backspace
            if chars:
                chars.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        chars.append(ch)
        sys.stdout.write("*")
        sys.stdout.flush()


def _prompt_pat() -> str:
    prompt = "> 请粘贴 PAT (pt- 开头, 逐字显示为*; 直接回车结束): "
    if os.name == "nt":
        return _read_masked(prompt)
    return getpass.getpass(prompt)


def verify_pat(pat: str) -> str:
    """真验证: PAT → jobToken 冷交换。返回昵称; 失败抛异常。

    只读冷交换, 不轮换任何在线会话 (在线桥持有的是自己的 refreshToken 链,
    与一次性的 jobToken 交换互不影响)。
    """
    import asyncio

    async def _go():
        return await qoder_auth.exchange_job_token(
            pat, _machine_id(), _machine_token(), _machine_type()
        )

    jt = asyncio.run(_go())
    name = str(jt.get("name") or "").strip()
    if not name:
        raise RuntimeError("网关返回缺少昵称, 视为异常")
    return name


def _machine_id() -> str:
    import uuid

    return str(uuid.uuid4())


def _machine_token() -> str:
    import base64
    import uuid

    raw = (uuid.uuid4().hex + uuid.uuid4().hex)[:50]
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _machine_type() -> str:
    import uuid

    return uuid.uuid4().hex[:18]


def service_running() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(SERVICE_URL + "/status", timeout=3):
            return True
    except Exception:
        return False


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("  Qoder 账号池添加工具 (pool.json)")
    print("=" * 60)

    args = [a.strip() for a in sys.argv[1:] if a.strip()]
    cfg, existing, source = _read_pool_or_fallback()
    print(f"当前池内账号: {len(existing)} 个 (来源 {source})")
    print("输入 PAT 后立即向网关验证, 通过才写入。Ctrl+C 取消。\n")

    def inputs():
        """账号来源三分支: 命令行参数 → 管道 stdin → 交互 (静默回显)。

        注意: Windows 的 getpass 直接读 CONIN$ 不认 stdin 重定向, 非 tty 时
        必须先走 stdin 逐行读取, 否则会挂在控制台等待上。
        """
        if args:
            yield from args
            return
        if not sys.stdin or not sys.stdin.isatty():
            for line in sys.stdin or []:
                line = line.strip()
                if line:
                    yield line
            return
        while True:
            try:
                pat = _prompt_pat().strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not pat:
                return
            yield pat

    added: list[str] = []
    from_args = bool(args)
    for pat in inputs():
        shown = f"> 待添加{' (命令行参数)' if from_args else ''}: {pat[:8]}...{pat[-4:]}"
        print(shown)
        if not pat.startswith("pt-"):
            print("  ✗ 格式不对: PAT 应以 pt- 开头, 未写入。")
            continue
        if pat in existing or pat in added:
            print("  ! 该 PAT 已在池中, 跳过。")
            continue
        try:
            name = verify_pat(pat)
        except Exception as e:
            print(f"  ✗ 网关验证失败, 未写入: {str(e)[:200]}")
            continue
        added.append(pat)
        print(f"  ✓ 验证通过: {name} (尾号 ...{pat[-4:]}), 已加入待写清单")

    if not added:
        print("\n没有新增账号, 配置文件未改动。")
        return

    final_pats = existing + added
    new_cfg = dict(cfg)  # 保留 options / 其他字段
    new_cfg["pats"] = final_pats
    key_note = ""
    if not str(new_cfg.get("gateway_key") or "").strip():
        new_cfg["gateway_key"] = "gk-" + secrets.token_hex(16)
        key_note = (
            f"\n[add-pat] 已生成网关统一 key: {new_cfg['gateway_key']}\n"
            f"          客户端(如 Hermes)的 api_key 填它即可走账号池; 填 pt-xxx 则直通指定账号。"
        )

    tmp = POOL_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, POOL_PATH)  # 原子替换: 服务侧 stat 指纹必然捕捉到变化

    print(f"\n[add-pat] 已写入 pool.json: 共 {len(final_pats)} 个账号 (新增 {len(added)} 个)")
    if key_note:
        print(key_note)
    if service_running():
        print("[add-pat] 检测到服务在跑 —— 账号池热加载, 下一个请求/签到周期自动生效, 无需重启。")
    else:
        print("[add-pat] 服务当前未运行; 启动后新账号自然生效。")


if __name__ == "__main__":
    main()

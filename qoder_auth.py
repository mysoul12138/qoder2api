"""
qoder_auth — Qoder 网关认证与请求客户端（单文件整合）。

合并自原 signature.py / qoder_encoding.py / config.py / bearer_builder.py /
signature_api_client.py / bearer_api_client.py，对应 Java 版
SignatureApiClient + JobTokenClient + BearerBuilder + BearerApiClient。

分层（自底向上）：
    1. 区域配置          —— API 端点与 URL 构造
    2. 自定义编码        —— Qoder 专用 Base64 变种
    3. 请求签名          —— MD5 签名 + RFC1123 日期
    4. Bearer 构建       —— RSA/AES 加密 + AuthIdentity / SessionContext
    5. 签名 API 客户端   —— PAT→jobToken / 用户状态 / 心跳
    6. Bearer API 客户端 —— 带 Bearer 的 HTTP + SSE 流
"""

import base64
import hashlib
import json
import os
import platform
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from email.utils import formatdate
from urllib.parse import urlparse

import httpx
from Crypto.Cipher import AES, PKCS1_v1_5
from Crypto.PublicKey import RSA


# ════════════════════════════════════════════════════════════════════════
# 1. 区域配置 (原 config.py)
# ════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RegionConfig:
    """某个区域的全部 API 端点配置。"""

    name: str  # 区域标识
    auth_base: str  # 认证类接口基地址 (jobToken / status / heartbeat)
    chat_base: str  # 对话类接口基地址 (agent_chat_generation)


CN = RegionConfig(
    name="cn",
    auth_base="https://gateway.qoder.com.cn",
    chat_base="https://gateway.qoder.com.cn",
)


def resolve(pat: str) -> tuple[str, RegionConfig]:
    """直接返回 PAT 和国内版区域配置。"""
    return pat, CN


def auth_url(region: RegionConfig, path: str) -> str:
    """构造认证类 URL。path 示例: '/algo/api/v3/user/jobToken?Encode=1'"""
    return f"{region.auth_base}{path}"


def chat_url(region: RegionConfig) -> str:
    """构造对话 SSE URL。"""
    return (
        f"{region.chat_base}/algo/api/v2/service/pro/sse/agent_chat_generation"
        "?FetchKeys=llm_model_result&AgentId=agent_common&Encode=1"
    )


def model_list_url(region: RegionConfig) -> str:
    """构造模型目录 URL (与 CLI 的 /api/v2/model/list?Encode=1 同源)。"""
    return f"{region.chat_base}/algo/api/v2/model/list?Encode=1"


async def fetch_model_catalog(sess: "SessionContext", region: RegionConfig) -> dict:
    """GET 模型目录, 返回解析后的 JSON。

    响应为明文 JSON (实测与 chat SSE 端点一致的 Encode=1 行为)。若服务端
    日后改为整包加密, resp.json() 会抛错, 由调用方 (OpenAiBridge.get_catalog)
    捕获并回退到内置兜底表。
    """
    return await call_get(sess, model_list_url(region))


# ════════════════════════════════════════════════════════════════════════
# 2. 自定义编码 (原 qoder_encoding.py)
# ════════════════════════════════════════════════════════════════════════
# 自定义 Base64：私有字母表（含 @#&*()^.! 等）+ 字符重排。

CUSTOM_ALPHABET = "_doRTgHZBKcGVjlvpC,@aFSx#DPuNJme&i*MzLOEn)sUrthbf%Y^w.(kIQyXqWA!"
STD_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
CUSTOM_PAD = "$"

# 构建映射表
_C2S: dict[str, str] = {}
_S2C: dict[str, str] = {}

for _i in range(64):
    _C2S[CUSTOM_ALPHABET[_i]] = STD_ALPHABET[_i]
    _S2C[STD_ALPHABET[_i]] = CUSTOM_ALPHABET[_i]

_C2S[CUSTOM_PAD] = "="
_S2C["="] = CUSTOM_PAD


def encode(plaintext: bytes) -> str:
    """将明文字节进行自定义编码。"""
    std = base64.b64encode(plaintext).decode("ascii")
    n = len(std)
    a = n // 3
    rearranged = std[n - a :] + std[a : n - a] + std[:a]
    result = []
    for ch in rearranged:
        m = _S2C.get(ch)
        if m is None:
            raise ValueError(f"char out of alphabet: {ch!r}")
        result.append(m)
    return "".join(result)


def decode(encoded: str) -> bytes:
    """将自定义编码字符串解码回明文字节。"""
    n = len(encoded)
    mapped = []
    for ch in encoded:
        m = _C2S.get(ch)
        if m is None:
            raise ValueError(f"char out of custom alphabet: {ch!r}")
        mapped.append(m)
    mapped_str = "".join(mapped)
    a = n // 3
    std = mapped_str[n - a :] + mapped_str[a : n - a] + mapped_str[:a]
    return base64.b64decode(std)


# ════════════════════════════════════════════════════════════════════════
# 3. 请求签名 (原 signature.py)
# ════════════════════════════════════════════════════════════════════════
# 生成 MD5 签名和 RFC1123 日期。

APPCODE = "cosy"
_DEFAULT_SECRET = "d2FyLCB3YXIgbmV2ZXIgY2hhbmdlcw=="  # base64("war, war never changes")
SECRET = os.environ.get("QODER_SIGNATURE_SECRET", _DEFAULT_SECRET)
SEP = "&"


def current_date() -> str:
    """返回 RFC1123 格式的当前 UTC 时间。"""
    return formatdate(timeval=time.time(), localtime=False, usegmt=True)


def sign(date: str) -> str:
    """根据日期生成 MD5 签名。"""
    s = f"{APPCODE}{SEP}{SECRET}{SEP}{date}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════════════════════════
# 4. Bearer 构建：加密 + 数据类 (原 bearer_builder.py)
# ════════════════════════════════════════════════════════════════════════

SERVER_PUBKEY_PEM = (
    "-----BEGIN PUBLIC KEY-----\n"
    "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDA8iMH5c02LilrsERw9t6Pv5Nc\n"
    "4k6Pz1EaDicBMpdpxKduSZu5OANqUq8er4GM95omAGIOPOh+Nx0spthYA2BqGz+l\n"
    "6HRkPJ7S236FZz73In/KVuLnwI8JJ2CbuJap8kvheCCZpmAWpb/cPx/3Vr/J6I17\n"
    "XcW+ML9FoCI6AOvOzwIDAQAB\n"
    "-----END PUBLIC KEY-----"
)


@dataclass(frozen=True)
class AuthIdentity:
    name: str
    aid: str
    uid: str
    yx_uid: str
    organization_id: str
    organization_name: str
    user_type: str
    security_oauth_token: str
    refresh_token: str


@dataclass(frozen=True)
class SessionContext:
    temp_key: bytes
    cosy_key: str
    info: str
    identity: AuthIdentity
    machine_id: str
    machine_token: str
    machine_type: str


def _rsa_encrypt(temp_key: bytes) -> bytes:
    key = RSA.import_key(SERVER_PUBKEY_PEM)
    cipher = PKCS1_v1_5.new(key)
    return cipher.encrypt(temp_key)


def _aes_encrypt(plain: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv=key)
    # PKCS7 padding
    pad_len = 16 - (len(plain) % 16)
    padded = plain + bytes([pad_len] * pad_len)
    return cipher.encrypt(padded)


def _md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _auth_payload_json(identity: AuthIdentity) -> bytes:
    payload = {
        "name": identity.name,
        "aid": identity.aid,
        "uid": identity.uid,
        "yx_uid": identity.yx_uid,
        "organization_id": identity.organization_id,
        "organization_name": identity.organization_name,
        "user_type": identity.user_type,
        "security_oauth_token": identity.security_oauth_token,
        "refresh_token": identity.refresh_token,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def new_session(
    identity: AuthIdentity, machine_id: str, machine_token: str, machine_type: str
) -> SessionContext:
    temp_key = uuid.uuid4().hex[:16].encode("ascii")
    cosy_key = base64.b64encode(_rsa_encrypt(temp_key)).decode("ascii")
    info = base64.b64encode(
        _aes_encrypt(_auth_payload_json(identity), temp_key)
    ).decode("ascii")
    return SessionContext(
        temp_key=temp_key,
        cosy_key=cosy_key,
        info=info,
        identity=identity,
        machine_id=machine_id,
        machine_token=machine_token,
        machine_type=machine_type,
    )


def sign_request(
    payload_b64: str, cosy_key: str, cosy_date: str, body: str, path_without_algo: str
) -> str:
    s = f"{payload_b64}\n{cosy_key}\n{cosy_date}\n{body}\n{path_without_algo}"
    return _md5_hex(s)


def build_payload_b64(info: str) -> str:
    m = {
        "cosyVersion": "0.1.43",
        "ideVersion": "",
        "info": info,
        "requestId": str(uuid.uuid4()),
        "version": "v1",
    }
    sorted_m = dict(sorted(m.items()))
    return base64.b64encode(
        json.dumps(sorted_m, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")


def compose_bearer(payload_b64: str, sig: str) -> str:
    return f"Bearer COSY.{payload_b64}.{sig}"


# ════════════════════════════════════════════════════════════════════════
# 5. 签名 API 客户端 (原 signature_api_client.py)
#    PAT → jobToken 交换 / 用户状态 / 心跳。使用 Signature 签名的 HTTP 请求。
# ════════════════════════════════════════════════════════════════════════


class QoderAuthError(RuntimeError):
    """Raised when the Qoder gateway rejects authentication (HTTP 401/403).

    Carries the status code so callers can decide whether to attempt a
    refresh-and-retry vs. propagate the failure.
    """

    def __init__(self, status_code: int, detail: str = ""):
        super().__init__(f"HTTP {status_code} {detail}".strip())
        self.status_code = status_code
        self.detail = detail


class QoderBusyError(RuntimeError):
    """上游表示"忙 / 排队 / 暂不可用"（例如 code 10605），而不是鉴权失败。

    为什么单列一类: 旧实现把带 403 信封的忙信号当成鉴权失败 → 白刷新一次会话
    （还会轮换 token，连累其他在途请求），重试后依旧失败，最后以 500 收尾。
    这类错误刷新无用，应让上层稍后重试或直接如实告诉客户端。
    """

    def __init__(self, detail: str = "", retry_after_seconds: int | None = None):
        message = f"upstream busy: {detail}" if detail else "upstream busy"
        if retry_after_seconds is not None:
            message = f"{message} (retry after {retry_after_seconds}s)"
        super().__init__(message)
        self.detail = detail
        self.retry_after_seconds = retry_after_seconds


def _common_headers(
    machine_id: str, machine_token: str, machine_type: str, date: str, sig: str
) -> dict:
    return {
        "cosy-machinetoken": machine_token,
        "cosy-machinetype": machine_type,
        "login-version": "v2",
        "appcode": APPCODE,
        "accept": "application/json",
        "accept-encoding": "identity",
        "cosy-version": "0.1.43",
        "cosy-clienttype": "5",
        "date": date,
        "signature": sig,
        "content-type": "application/json",
        "cosy-machineid": machine_id,
        "user-agent": "Go-http-client/2.0",
    }


async def _post_encoded(
    url: str, obj: dict, machine_id: str, machine_token: str, machine_type: str
) -> dict:
    date = current_date()
    sig = sign(date)
    plain = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    body = encode(plain)
    headers = _common_headers(machine_id, machine_token, machine_type, date, sig)

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, content=body.encode("utf-8"), headers=headers)
        if resp.status_code != 200:
            detail = resp.text[:300]
            if resp.status_code in (401, 403):
                raise QoderAuthError(resp.status_code, detail)
            raise RuntimeError(f"HTTP {resp.status_code} at {url} body={detail}")
        return resp.json()


async def _request_job_token(
    personal_token: str,
    refresh_token: str,
    security_oauth_token: str,
    need_refresh: bool,
    machine_id: str,
    machine_token: str,
    machine_type: str,
    region: RegionConfig,
) -> dict:
    """Shared helper for the PAT→jobToken and refresh flows.

    The gateway always requires ``personalToken`` (a PAT) on this endpoint; the
    ``refreshToken`` + ``securityOauthToken`` + ``needRefresh`` triplet selects
    the renewal path that rotates the short-lived session token instead of a
    cold re-exchange.
    """
    url = auth_url(region, "/algo/api/v3/user/jobToken?Encode=1")
    inner = {
        "personalToken": personal_token,
        "securityOauthToken": security_oauth_token,
        "refreshToken": refresh_token,
        "needRefresh": need_refresh,
        "authInfo": {},
    }
    outer = {
        "payload": json.dumps(inner, separators=(",", ":")),
        "encodeVersion": "1",
    }
    return await _post_encoded(url, outer, machine_id, machine_token, machine_type)


async def exchange_job_token(
    personal_token: str,
    machine_id: str,
    machine_token: str,
    machine_type: str,
    region: RegionConfig | None = None,
) -> dict:
    """Cold PAT → jobToken exchange (needRefresh=False)."""
    region = region or CN
    return await _request_job_token(
        personal_token=personal_token,
        refresh_token="",
        security_oauth_token="",
        need_refresh=False,
        machine_id=machine_id,
        machine_token=machine_token,
        machine_type=machine_type,
        region=region,
    )


async def refresh_job_token(
    personal_token: str,
    refresh_token: str,
    security_oauth_token: str,
    machine_id: str,
    machine_token: str,
    machine_type: str,
    region: RegionConfig | None = None,
) -> dict:
    """Renew the short-lived session token via the stored refreshToken.

    Sends the previously issued ``refreshToken`` + ``securityOauthToken`` with
    ``needRefresh=True`` to rotate the 24h session token (and receive a fresh
    refreshToken). The PAT is still required by the gateway, but this is a
    graceful renewal rather than a cold re-exchange.
    """
    region = region or CN
    return await _request_job_token(
        personal_token=personal_token,
        refresh_token=refresh_token,
        security_oauth_token=security_oauth_token,
        need_refresh=True,
        machine_id=machine_id,
        machine_token=machine_token,
        machine_type=machine_type,
        region=region,
    )


async def user_status(
    user_id: str,
    machine_id: str,
    machine_token: str,
    machine_type: str,
    region: RegionConfig | None = None,
) -> dict:
    region = region or CN
    url = auth_url(region, "/algo/api/v3/user/status?Encode=1")
    inner = {
        "userId": user_id,
        "personalToken": "",
        "securityOauthToken": "",
        "refreshToken": "",
        "needRefresh": False,
        "authInfo": {},
    }
    outer = {
        "payload": json.dumps(inner, separators=(",", ":")),
        "encodeVersion": "1",
    }
    return await _post_encoded(url, outer, machine_id, machine_token, machine_type)


async def heartbeat(
    machine_id: str,
    machine_token: str,
    machine_type: str,
    region: RegionConfig | None = None,
) -> dict:
    region = region or CN
    url = auth_url(region, "/algo/api/v1/heartbeat?Encode=1")
    arch = platform.machine()
    os_arch = "windows_amd64" if arch in ("AMD64", "x86_64") else arch
    os_version = f"{platform.system()} {platform.release()}"
    hb = {
        "event_time": int(time.time() * 1000),
        "event_type": "cosy_heartbeat",
        "mid": machine_id,
        "os_arch": os_arch,
        "os_version": os_version,
        "ide_type": "qodercli",
        "ide_version": "0.1.43",
        "extra_info": {},
    }
    return await _post_encoded(url, hb, machine_id, machine_token, machine_type)


# ════════════════════════════════════════════════════════════════════════
# 6. Bearer API 客户端 (原 bearer_api_client.py)
#    使用 Bearer Token 认证的 HTTP 请求 + SSE 流。
# ════════════════════════════════════════════════════════════════════════


def _make_common_headers(
    sess: SessionContext, date: str, bearer: str, accept: str
) -> dict:
    """构建所有请求共享的公共 headers。"""
    return {
        "cosy-data-policy": "AGREE",
        "content-type": "application/json",
        "cosy-machinetype": sess.machine_type,
        "cosy-clienttype": "5",
        "cosy-date": date,
        "cosy-user": sess.identity.uid,
        "cosy-key": sess.cosy_key,
        "accept": accept,
        "authorization": bearer,
        "accept-encoding": "identity",
        "cosy-version": "0.1.43",
        "cosy-machineid": sess.machine_id,
        "cosy-machinetoken": sess.machine_token,
        "login-version": "v2",
        "user-agent": "Go-http-client/2.0",
    }


def _build_bearer(sess: SessionContext, date: str, body: str, path_sig: str) -> str:
    """构建一次请求的 Bearer token。"""
    payload_b64 = build_payload_b64(sess.info)
    sig = sign_request(payload_b64, sess.cosy_key, date, body, path_sig)
    return compose_bearer(payload_b64, sig)


def _sig_path(full_url: str) -> str:
    u = urlparse(full_url)
    path = u.path
    if path.startswith("/algo"):
        path = path[len("/algo") :]
    return path


async def call_post(sess: SessionContext, full_url: str, json_body: dict) -> dict:
    return await _call(sess, "POST", full_url, json_body, None)


async def call_get(sess: SessionContext, full_url: str) -> dict:
    return await _call(sess, "GET", full_url, None, None)


async def _call(
    sess: SessionContext,
    method: str,
    full_url: str,
    json_body: dict | None,
    extra_headers: dict | None,
) -> dict:
    path_sig = _sig_path(full_url)
    body = ""
    if json_body is not None:
        plain = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
        body = encode(plain)

    date = str(int(time.time()))
    bearer = _build_bearer(sess, date, body, path_sig)
    headers = _make_common_headers(sess, date, bearer, "application/json")
    if extra_headers:
        headers.update(extra_headers)

    async with httpx.AsyncClient(timeout=30) as client:
        if method == "POST":
            resp = await client.post(
                full_url, content=body.encode("utf-8"), headers=headers
            )
        else:
            resp = await client.get(full_url, headers=headers)
        if resp.status_code != 200:
            detail = resp.text[:300]
            busy = _busy_from_text(detail)
            if busy is not None:
                raise QoderBusyError(*busy)
            if resp.status_code in (401, 403):
                raise QoderAuthError(resp.status_code, detail)
            raise RuntimeError(f"HTTP {resp.status_code} body={detail}")
        return resp.json()


async def open_stream_lines(
    sess: SessionContext,
    full_url: str,
    json_body: dict,
    extra_headers: dict | None,
) -> AsyncIterator[str]:
    """发送 POST 请求并以 SSE 方式逐行产出响应 (异步生成器)。"""
    path_sig = _sig_path(full_url)
    body = encode(json.dumps(json_body, separators=(",", ":")).encode("utf-8"))
    date = str(int(time.time()))
    bearer = _build_bearer(sess, date, body, path_sig)
    headers = _make_common_headers(sess, date, bearer, "text/event-stream")
    headers["cache-control"] = "no-cache"
    if extra_headers:
        headers.update(extra_headers)

    timeout = httpx.Timeout(connect=15, read=300, write=15, pool=15)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", full_url, content=body.encode("utf-8"), headers=headers
        ) as resp:
            if resp.status_code != 200:
                err_body = (await resp.aread()).decode("utf-8")[:300]
                busy = _busy_from_text(err_body)
                if busy is not None:
                    raise QoderBusyError(*busy)
                if resp.status_code in (401, 403):
                    raise QoderAuthError(resp.status_code, err_body)
                raise RuntimeError(f"HTTP {resp.status_code} {err_body}")

            async for line in resp.aiter_lines():
                if not line:
                    continue
                is_busy, busy_detail, retry_after = _detect_busy_line(line)
                if is_busy:
                    # 忙/排队不是鉴权失败: 绝不刷新会话(刷新只会白轮换 token,
                    # 还可能连累其他在途请求), 直接上抛给调用方稍后重试。
                    raise QoderBusyError(busy_detail, retry_after)
                is_auth_err, detail = _detect_in_stream_auth_error(line)
                if is_auth_err:
                    raise QoderAuthError(401, detail)
                yield line

    print("[stream] read complete")


# 上游"忙 / 排队 / 暂不可用"信号: 这类信封也常带 403, 但刷新会话毫无意义。
# 实测(2026-09-21)样本:
#   {"statusCodeValue":403,"body":"{\"code\":\"10605\",\"message\":
#     \"{\\\"isQueued\\\":false,\\\"retryAfterSeconds\\\":5,...}\"}"}
_BUSY_CODES = frozenset({"10605"})
_BUSY_MARKERS = ("isQueued", "serviceAvailable", "retryAfterSeconds", "waitTime", "queueType")
# 信封最多下钻几层。实测有双层形态（外层 code "403" → 内层 code "10605" → 队列详情,
# retryAfterSeconds 在第二层的 message 里）；上限给到 6 留余量，同时防止异常的超深
# 嵌套拖垮解析。
_BUSY_DRILL_LIMIT = 6


def _busy_body_of(obj: dict) -> dict | None:
    """取信封里的业务 body（也兼容没有信封包裹的裸 body）。"""
    body = _parse_body(obj)
    if isinstance(body, dict):
        return body
    if "code" in obj or "message" in obj:
        return obj
    return None


def _json_object_or_none(raw) -> dict | None:
    """字符串能按 JSON 解析成对象就返回它，否则 None。"""
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _busy_payload_of(body: dict) -> tuple[dict | None, str]:
    """逐层下钻信封，返回 (含忙标记的元数据层, 该层对应的业务 code)。

    旧实现只挖一层 message: 遇到"外层 403 套内层 10605"的双层信封时,
    拿到的只是内层信封本身, retryAfterSeconds 还埋在它的 message 里 ——
    结果 503 丢掉了 Retry-After。这里按 message 逐层下钻（有层数上限）,
    停在第一个带 _BUSY_MARKERS 的层; code 取沿途最深的那个业务码。
    """
    current = body
    code = str(body.get("code") or "")
    for _ in range(_BUSY_DRILL_LIMIT):
        payload = _json_object_or_none(current.get("message"))
        if payload is None:
            break
        if payload.get("code") not in (None, ""):
            code = str(payload["code"])  # 越深越贴近业务层
        if any(key in payload for key in _BUSY_MARKERS):
            return payload, code
        current = payload  # 这一层仍是纯信封, 继续下钻
    # 兜底: 没有 message 可挖时, body 本身可能就是元数据层（或至少带忙业务码）。
    if any(key in body for key in _BUSY_MARKERS):
        return body, code
    if str(body.get("code") or "") in _BUSY_CODES:
        return {}, code
    return None, code


def detect_upstream_busy(obj: dict) -> tuple[bool, str, int | None]:
    """识别"上游忙 / 排队"。返回 (是否忙, 说明, 建议重试秒数)。"""
    if not isinstance(obj, dict):
        return False, "", None
    body = _busy_body_of(obj)
    if body is None:
        return False, "", None
    payload, code = _busy_payload_of(body)
    if payload is None:
        return False, "", None

    detail = " ".join(
        part
        for part in (code, json.dumps(payload, ensure_ascii=False) if payload else "")
        if part
    )
    raw_retry = payload.get("retryAfterSeconds")
    retry_after = (
        max(0, int(raw_retry))
        if isinstance(raw_retry, (int, float)) and not isinstance(raw_retry, bool)
        else None
    )
    return True, detail or "busy", retry_after


def _detect_busy_line(line: str) -> tuple[bool, str, int | None]:
    """从 SSE 行里识别忙/排队信号。"""
    s = line.strip()
    if not s.startswith("data:"):
        return False, "", None
    try:
        obj = json.loads(s[5:].strip())
    except (json.JSONDecodeError, ValueError):
        return False, "", None
    if not isinstance(obj, dict):
        return False, "", None
    return detect_upstream_busy(obj)


def _busy_from_text(text: str) -> tuple[str, int | None] | None:
    """从原始响应文本里识别忙/排队信号；命中返回 (说明, 建议重试秒数)。"""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    is_busy, detail, retry_after = detect_upstream_busy(obj)
    return (detail, retry_after) if is_busy else None


def _detect_in_stream_auth_error(line: str) -> tuple[bool, str]:
    """The Qoder gateway returns HTTP 200 but signals auth failure inside the
    SSE stream as an envelope like:
        data:{"body":"{\"code\":\"105\",\"message\":\"Login expired\"}",
              "statusCodeValue":403,"statusCode":"FORBIDDEN"}
    Detect that and surface it as an auth error so callers can refresh+retry.

    注意: 带 403 信封的"忙 / 排队"信号（如 code 10605）必须先排除 —— 它不是
    鉴权失败，刷新会话没有用（只会白轮换 token）。
    """
    s = line.strip()
    if not s.startswith("data:"):
        return False, ""
    try:
        obj = json.loads(s[5:].strip())
    except (json.JSONDecodeError, ValueError):
        return False, ""
    if not isinstance(obj, dict):
        return False, ""

    if detect_upstream_busy(obj)[0]:
        return False, ""

    scv = obj.get("statusCodeValue")
    if scv in (401, 403):
        return True, f"{scv} {_extract_body_message(obj)}"

    # also catch code 105 ("Login expired") nested in the body string
    msg = _extract_body_message(obj)
    body_obj = _parse_body(obj)
    if isinstance(body_obj, dict) and body_obj.get("code") in ("105", 105):
        return True, msg or "Login expired"
    return False, ""


def _parse_body(obj: dict):
    body = obj.get("body")
    if isinstance(body, (dict, list)):
        return body
    if isinstance(body, str):
        try:
            return json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _extract_body_message(obj: dict) -> str:
    body_obj = _parse_body(obj)
    if isinstance(body_obj, dict):
        return str(body_obj.get("message") or body_obj.get("code") or "")
    return ""

"""OpenAI ↔ Qoder 消息转换 + 流式累积器。

纯函数层,不依赖 flask / qoder_auth / 网络。
从原 openai_bridge.py 拆出。"""

from collections.abc import Callable
from dataclasses import dataclass
import copy
import json


@dataclass
class BridgeDelta:
    role: str = ""
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list | None = None

    def is_empty(self) -> bool:
        return (
            not self.role
            and not self.content
            and not self.reasoning_content
            and (self.tool_calls is None or len(self.tool_calls) == 0)
        )


class ToolCallAccumulator:
    """累积来自多个 delta 的 tool_calls。"""

    def __init__(self):
        self.calls: list[dict] = []

    def append(self, delta_calls: list):
        for dc in delta_calls:
            idx = (
                dc.get("index", len(self.calls))
                if isinstance(dc.get("index"), int)
                else len(self.calls)
            )
            while len(self.calls) <= idx:
                self.calls.append(
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )
            existing = self.calls[idx]
            if isinstance(dc.get("id"), str):
                existing["id"] = dc["id"]
            if isinstance(dc.get("type"), str):
                existing["type"] = dc["type"]
            df = dc.get("function", {})
            ef = existing["function"]
            if isinstance(df.get("name"), str):
                ef["name"] = df["name"]
            if isinstance(df.get("arguments"), str):
                ef["arguments"] = ef["arguments"] + df["arguments"]

    def is_empty(self) -> bool:
        return len(self.calls) == 0

    def snapshot(self) -> list:
        return copy.deepcopy(self.calls)


class StreamAccumulator:
    """流式响应的内容累积器。"""

    def __init__(
        self,
        req_id: str,
        created: int,
        model: str,
        tool_call_fallback: bool,
        emit_fn: Callable[[str], None] | None = None,
    ):
        self.req_id = req_id
        self.created = created
        self.model = model
        self.tool_call_fallback = tool_call_fallback
        self.tool_calls = ToolCallAccumulator()
        self.pending_content: list[str] = []
        self.pending_role = "assistant"
        self.emitted = False
        self.streaming_text = False
        self._emit_fn = emit_fn
        self._chunks: list[str] = []  # 仅在无 emit_fn 时使用（兼容非实时场景）

    def accept(self, delta: BridgeDelta):
        if delta.role:
            self.pending_role = delta.role

        if delta.reasoning_content:
            self._emit(None, delta.reasoning_content, None)

        # tool_calls 优先处理
        if delta.tool_calls and len(delta.tool_calls) > 0:
            self._discard_buffered_tool_call_text()
            self.tool_calls.append(delta.tool_calls)
            self._emit(None, None, self._with_tool_call_indices(delta.tool_calls))
            return

        if not delta.content:
            return

        if not self.tool_call_fallback or self.streaming_text:
            self.streaming_text = True
            self._emit(delta.content, None, None)
            return

        self.pending_content.append(delta.content)
        text = "".join(self.pending_content)
        if self._is_potential_tool_call_text(text):
            return
        self.streaming_text = True
        self._emit_buffered_text()

    def flush(self):
        if not self.pending_content:
            return
        buffered = "".join(self.pending_content)
        self.pending_content.clear()
        parsed = parse_tool_calls_text(buffered) if self.tool_call_fallback else None
        if parsed is not None:
            self.tool_calls.append(parsed)
            self._emit(None, None, self._with_tool_call_indices(parsed))
            return
        self.streaming_text = True
        self._emit(buffered, None, None)

    def finish_reason(self) -> str:
        return "tool_calls" if not self.tool_calls.is_empty() else "stop"

    def get_chunks(self) -> list[str]:
        return self._chunks

    def _emit_buffered_text(self):
        if not self.pending_content:
            return
        buffered = "".join(self.pending_content)
        self.pending_content.clear()
        self._emit(buffered, None, None)

    def _discard_buffered_tool_call_text(self):
        if not self.pending_content:
            return
        buffered = "".join(self.pending_content)
        self.pending_content.clear()
        if self.tool_call_fallback and self._is_potential_tool_call_text(buffered):
            return
        self.streaming_text = True
        self._emit(buffered, None, None)

    def _emit(
        self,
        content: str | None,
        reasoning_content: str | None,
        tool_calls: list | None,
    ):
        role = ""
        if not self.emitted:
            role = self.pending_role or "assistant"
        chunk = _make_sse_chunk(
            self.req_id,
            self.created,
            self.model,
            role,
            content,
            reasoning_content,
            tool_calls,
        )
        if self._emit_fn is not None:
            self._emit_fn(chunk)
        else:
            self._chunks.append(chunk)
        self.emitted = True

    @staticmethod
    def _is_potential_tool_call_text(text: str) -> bool:
        candidate = text.lstrip()
        if not candidate:
            return True
        return "Tool calls:".startswith(candidate) or candidate.startswith(
            "Tool calls:"
        )

    @staticmethod
    def _with_tool_call_indices(raw_tool_calls: list) -> list:
        indexed = []
        for i, tc in enumerate(raw_tool_calls):
            call = copy.deepcopy(tc)
            if not isinstance(call.get("index"), int):
                call["index"] = i
            indexed.append(call)
        return indexed


def _normalize_content(content) -> str:
    """将各种 content 格式统一为纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            part = _normalize_content_part(item)
            if part:
                parts.append(part)
        return "\n\n".join(parts)
    if isinstance(content, dict):
        return _normalize_content_part(content)
    return str(content)


def _normalize_content_part(item) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        t = item.get("type", "")
        if isinstance(item.get("text"), str):
            return item["text"]
        if t in ("image_url", "input_image"):
            url = item.get("image_url", {}).get("url", "")
            if url:
                return f"[image] {url}"
        if isinstance(item.get("content"), (list, dict)):
            return _normalize_content(item["content"])
        return json.dumps(item, ensure_ascii=False)
    return str(item)


def _extract_image_data_url(part: dict) -> str | None:
    """Pull a data: URL (or http URL) out of an OpenAI image content part.

    Accepts both ``image_url`` (``{type:'image_url', image_url:{url}}``) and
    the newer ``input_image`` (``{type:'input_image', image_url:{url}}`` /
    ``{type:'input_image', image_url:{url}}``) shapes.
    """
    if not isinstance(part, dict):
        return None
    t = part.get("type", "")
    if t not in ("image_url", "input_image"):
        return None
    # image_url may be a dict {url, detail} or, in some clients, a string
    iu = part.get("image_url")
    if isinstance(iu, dict):
        url = iu.get("url")
    elif isinstance(iu, str):
        url = iu
    else:
        url = part.get("url")
    return url if isinstance(url, str) and url else None


def _extract_message_images(message: dict) -> list[str]:
    """Return all image data/http URLs found in an OpenAI message, in order.

    Handles both the string-content and the array-of-parts content forms.
    """
    content = message.get("content")
    urls: list[str] = []
    if isinstance(content, list):
        for part in content:
            url = _extract_image_data_url(part) if isinstance(part, dict) else None
            if url:
                urls.append(url)
    return urls


def _normalize_message_text(message: dict) -> str:
    text = _normalize_content(message.get("content"))
    if not text.strip():
        text = _normalize_content(message.get("contents"))
    return text


def _normalize_tool_arguments(arguments) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def _normalize_tool_calls(raw_tool_calls) -> list | None:
    if not isinstance(raw_tool_calls, list):
        return None
    normalized = []
    for rtc in raw_tool_calls:
        func = rtc.get("function", {})
        name = func.get("name", "")
        arguments = _normalize_tool_arguments(func.get("arguments"))
        if not name and not arguments:
            continue
        normalized.append(
            {
                "id": rtc.get("id", ""),
                "type": rtc.get("type", "function"),
                "function": {"name": name, "arguments": arguments},
            }
        )
    return normalized if normalized else None


def parse_tool_calls_text(text: str | None) -> list | None:
    """尝试从文本中解析 'Tool calls: [...]' 格式的 tool_calls。"""
    if text is None:
        return None
    trimmed = text.strip()
    if not trimmed.startswith("Tool calls:"):
        return None
    payload = trimmed[len("Tool calls:") :].strip()
    if payload.startswith("```") and payload.endswith("```"):
        newline = payload.find("\n")
        if newline >= 0:
            payload = payload[newline + 1 : -3].strip()
    if not payload.startswith("["):
        return None
    try:
        parsed = json.loads(payload)
        return _normalize_tool_calls(parsed)
    except Exception:  # JSON 解析失败 / 结构不符 → 不是工具调用文本，返回 None
        return None


def _blank_response_meta() -> dict:
    return {
        "id": "",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }


def _build_user_message(text: str, images: list[str] | None = None) -> dict:
    # Qoder's chat gateway speaks OpenAI-style content parts in the `contents`
    # array. Images go in first as `{type:'image_url', image_url:{url}}` data
    # URLs (or http URLs), followed by the text part.
    parts: list[dict] = []
    for url in images or []:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    if text.strip():
        parts.append({"type": "text", "text": text})
    if not parts:
        parts.append({"type": "text", "text": ""})
    return {
        "role": "user",
        "content": "",
        "contents": parts,
        "response_meta": _blank_response_meta(),
        "reasoning_content_signature": "",
    }


def _build_structured_message(role: str, text: str | None) -> dict:
    return {
        "role": role,
        "content": text or "",
        "response_meta": _blank_response_meta(),
        "reasoning_content_signature": "",
    }


def _build_assistant_tool_call_message(text: str, tool_calls: list) -> dict:
    content = text or ""
    if parse_tool_calls_text(content) is not None:
        content = ""
    msg = _build_structured_message("assistant", content)
    msg["tool_calls"] = copy.deepcopy(tool_calls)
    return msg


def _build_tool_message(message: dict, text: str) -> dict:
    out = _build_structured_message("tool", text)
    if isinstance(message.get("name"), str):
        out["name"] = message["name"]
    if isinstance(message.get("tool_call_id"), str):
        out["tool_call_id"] = message["tool_call_id"]
    return out


def _render_tool_calls(tool_calls) -> str:
    return "Tool calls:\n" + json.dumps(tool_calls, ensure_ascii=False)


def _render_tool_result(message: dict, text: str) -> str:
    name = message.get("name", "")
    tool_call_id = message.get("tool_call_id", "")
    parts = ["Tool result"]
    if name:
        parts.append(f" ({name})")
    if tool_call_id:
        parts.append(f" [{tool_call_id}]")
    if text.strip():
        parts.append(f":\n{text}")
    return "".join(parts)


def _summarize_unresolved_tool_calls(tool_calls: list) -> str:
    sb = ["Previously planned but unexecuted tool calls"]
    limit = min(len(tool_calls), 6)
    names = []
    for i in range(limit):
        name = tool_calls[i].get("function", {}).get("name", "") or "unknown"
        names.append(name)
    if names:
        sb.append(": ")
        sb.append(", ".join(names))
    if len(tool_calls) > limit:
        sb.append(f" and {len(tool_calls) - limit} more")
    sb.append(".")
    return "".join(sb)


def _join_sections(first: str | None, second: str | None) -> str:
    if not first or not first.strip():
        return second or ""
    if not second or not second.strip():
        return first
    return first + "\n\n" + second


def _has_resolved_tool_response(messages: list, assistant_index: int) -> bool:
    message = messages[assistant_index]
    if message.get("role") != "assistant":
        return False
    tc = message.get("tool_calls")
    has_tool_calls = (isinstance(tc, list) and len(tc) > 0) or parse_tool_calls_text(
        _normalize_message_text(message)
    ) is not None
    if not has_tool_calls:
        return False
    for i in range(assistant_index + 1, len(messages)):
        next_role = messages[i].get("role", "")
        if next_role == "tool":
            return True
        if next_role in ("assistant", "user", "system"):
            return False
    return False


def _extract_any_tool_calls(
    message: dict, text: str, tools_enabled: bool
) -> list | None:
    if not tools_enabled:
        return None
    tc = message.get("tool_calls")
    if isinstance(tc, list) and len(tc) > 0:
        return _normalize_tool_calls(tc)
    return parse_tool_calls_text(text)


def _convert_incoming_message(
    message: dict, tools_enabled: bool, allow_structured_tool_calls: bool
) -> dict | None:
    role = message.get("role", "user")
    text = _normalize_message_text(message)
    any_tool_calls = _extract_any_tool_calls(message, text, tools_enabled)
    structured_tool_calls = None
    if tools_enabled and allow_structured_tool_calls:
        structured_tool_calls = _extract_any_tool_calls(message, text, True)

    if role == "assistant" and structured_tool_calls is not None:
        return _build_assistant_tool_call_message(text, structured_tool_calls)

    if (
        role == "assistant"
        and any_tool_calls is not None
        and not allow_structured_tool_calls
    ):
        return _build_structured_message(
            "assistant", _summarize_unresolved_tool_calls(any_tool_calls)
        )

    tc = message.get("tool_calls")
    if not tools_enabled and isinstance(tc, list) and len(tc) > 0:
        text = _join_sections(text, _render_tool_calls(tc))

    if role == "tool":
        if tools_enabled:
            return _build_tool_message(message, text)
        role = "user"
        text = _render_tool_result(message, text)

    # collect any inline images (only meaningful for user messages)
    images = _extract_message_images(message) if role == "user" else []
    # when images are carried as real image_url parts, drop the "[image] <url>"
    # textual fallback that _normalize_message_text injected, so we don't send
    # the same image twice (once as a part, once as text).
    if images:
        text = "\n\n".join(
            s
            for s in (seg.strip() for seg in text.split("\n\n"))
            if s and not s.startswith("[image] ")
        )

    if not text.strip() and not images:
        return None

    if role == "user":
        return _build_user_message(text, images)

    return _build_structured_message(role, text)


def _build_qoder_messages(
    incoming_messages: list, prompt: str, tools_enabled: bool
) -> list:
    rebuilt = []

    if incoming_messages:
        for i, message in enumerate(incoming_messages):
            allow_structured = tools_enabled and _has_resolved_tool_response(
                incoming_messages, i
            )
            converted = _convert_incoming_message(
                message, tools_enabled, allow_structured
            )
            if converted is not None:
                rebuilt.append(converted)

    if not rebuilt and prompt.strip():
        rebuilt.append(_build_user_message(prompt))

    return rebuilt


def _apply_openai_tool_config(body: dict, req_body: dict) -> bool:
    incoming_tools = req_body.get("tools")
    tools_enabled = isinstance(incoming_tools, list) and len(incoming_tools) > 0
    if tools_enabled:
        body["tools"] = copy.deepcopy(incoming_tools)
    else:
        body.pop("tools", None)
    if "tool_choice" in req_body:
        body["tool_choice"] = copy.deepcopy(req_body["tool_choice"])
    else:
        body.pop("tool_choice", None)
    if "parallel_tool_calls" in req_body:
        body["parallel_tool_calls"] = req_body["parallel_tool_calls"]
    else:
        body.pop("parallel_tool_calls", None)
    return tools_enabled


def _extract_latest_user_prompt(messages) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if message.get("role") == "user":
            text = _normalize_message_text(message)
            if text.strip():
                return text
    return ""


def _make_chunk(req_id: str, created: int, model: str) -> dict:
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }


def _make_sse_chunk(
    req_id: str,
    created: int,
    model: str,
    role: str | None,
    content: str | None,
    reasoning_content: str | None,
    tool_calls: list | None,
) -> str:
    chunk = _make_chunk(req_id, created, model)
    delta = chunk["choices"][0]["delta"]
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    if reasoning_content:
        delta["reasoning_content"] = reasoning_content
    if tool_calls and len(tool_calls) > 0:
        delta["tool_calls"] = tool_calls
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _extract_delta(data_line: str) -> BridgeDelta:
    try:
        wrapper = json.loads(data_line)
        inner_str = wrapper.get("body", "")
        if not inner_str:
            return BridgeDelta()
        inner_json = json.loads(inner_str)
        for ch in inner_json.get("choices", []):
            delta = ch.get("delta", {})
            role = delta.get("role", "")
            content = delta.get("content", "")
            reasoning_content = delta.get("reasoning_content", "")
            tc = delta.get("tool_calls")
            tool_calls = None
            if isinstance(tc, list) and len(tc) > 0:
                tool_calls = copy.deepcopy(tc)
            if role or content or reasoning_content or tool_calls is not None:
                return BridgeDelta(
                    role=role,
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls,
                )
    except Exception as e:
        # SSE 单行解析失败不应中断整条流；记一行便于排障，返回空 delta 跳过该行。
        # 注意：这里会吞掉非 JSON 行（如网关心跳 / 注释），属预期行为。
        print(
            f"[transform] _extract_delta skipped line: {e!r} data={data_line[:120]!r}"
        )
    return BridgeDelta()

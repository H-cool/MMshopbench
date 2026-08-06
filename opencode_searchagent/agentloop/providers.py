from __future__ import annotations

import http.client
import json
import mimetypes
import os
import socket
import time
from typing import Any, Callable, Protocol
from urllib.parse import urlparse
from urllib import error, request

from .types import UnifiedMessage, UnifiedRequest, UnifiedResponse, UnifiedToolCall


class ProviderAdapter(Protocol):
    model: str

    def complete(self, unified_request: UnifiedRequest) -> UnifiedResponse:
        ...


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 180.0,
        default_sampling: dict[str, Any] | None = None,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        strip_tool_message_name: bool | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_sampling = default_sampling or {}
        self.max_retries = max(0, max_retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.strip_tool_message_name = (
            _env_flag("AGENTLOOP_OPENAI_STRIP_TOOL_NAME", default=False)
            if strip_tool_message_name is None
            else strip_tool_message_name
        )
        self.last_raw_request: dict[str, Any] | None = None

    def complete(self, unified_request: UnifiedRequest) -> UnifiedResponse:
        payload = self._build_payload(unified_request)
        self.last_raw_request = payload
        raw = self._post_json("/chat/completions", payload)
        return self._parse_response(raw)

    def _build_payload(self, unified_request: UnifiedRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": unified_request.model or self.model,
            "messages": [
                self._message_to_openai(message.to_dict())
                for message in unified_request.messages
            ],
        }

        sampling = {**self.default_sampling, **unified_request.sampling}
        payload.update({key: value for key, value in sampling.items() if value is not None})

        if unified_request.tools:
            payload["tools"] = [
                {"type": "function", "function": tool}
                for tool in unified_request.tools
            ]
            payload["tool_choice"] = "auto"

        return payload

    def _message_to_openai(self, message: dict[str, Any]) -> dict[str, Any]:
        data = {key: value for key, value in message.items() if value is not None}
        if data.get("role") == "tool" and self.strip_tool_message_name:
            data.pop("name", None)
        return data

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST",
        )


        return _request_with_retries(
            req,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            reader=lambda resp: json.loads(resp.read().decode("utf-8")),
        )

    def _parse_response(self, raw: dict[str, Any]) -> UnifiedResponse:
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError("provider response has no choices")

        choice = choices[0]
        message = choice.get("message") or {}
        tool_calls = []

        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            if isinstance(arguments, str):
                try:
                    parsed_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed_arguments = {"_raw": arguments}
            elif isinstance(arguments, dict):
                parsed_arguments = arguments
            else:
                parsed_arguments = {"_raw": arguments}

            tool_calls.append(
                UnifiedToolCall(
                    id=str(call.get("id") or f"tool_call_{len(tool_calls)}"),
                    name=str(function.get("name") or ""),
                    arguments=parsed_arguments,
                    raw=call,
                )
            )

        return UnifiedResponse(
            text=message.get("content"),
            tool_calls=tool_calls,
            raw=raw,
            usage=raw.get("usage"),
            finish_reason=choice.get("finish_reason"),
        )


class GeminiVertexProvider:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: float = 180.0,
        default_sampling: dict[str, Any] | None = None,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_sampling = default_sampling or {}
        self.max_retries = max(0, max_retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.last_raw_request: dict[str, Any] | None = None

    def complete(self, unified_request: UnifiedRequest) -> UnifiedResponse:
        payload = self._build_payload(unified_request)
        self.last_raw_request = payload
        chunks = self._post_stream_json(
            self._stream_path(unified_request),
            payload,
            session_id=unified_request.session_id,
        )
        return self._parse_stream_response(chunks)

    def _stream_path(self, unified_request: UnifiedRequest) -> str:
        model = unified_request.model or self.model
        return f"/models/{model}:streamGenerateContent"

    def _build_payload(self, unified_request: UnifiedRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        contents: list[dict[str, Any]] = []
        system_parts: list[dict[str, Any]] = []

        index = 0
        while index < len(unified_request.messages):
            message = unified_request.messages[index]
            if message.role == "system":
                system_parts.extend(self._content_to_parts(message.content))
                index += 1
                continue
            if message.role == "tool":
                tool_parts = []
                while index < len(unified_request.messages):
                    tool_message = unified_request.messages[index]
                    if tool_message.role != "tool":
                        break
                    tool_parts.append(self._tool_response_part(tool_message))
                    index += 1
                contents.append({"role": "user", "parts": tool_parts})
                continue
            contents.append(self._message_to_gemini_content(message))
            index += 1

        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        if contents:
            payload["contents"] = contents

        sampling = {**self.default_sampling, **unified_request.sampling}
        generation_config = {
            key: value for key, value in sampling.items() if value is not None
        }
        if generation_config:
            payload["generationConfig"] = generation_config

        if unified_request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        self._tool_to_function_declaration(tool)
                        for tool in unified_request.tools
                    ]
                }
            ]

        return payload

    def _message_to_gemini_content(
        self,
        message: UnifiedMessage,
    ) -> dict[str, Any]:
        if message.role == "assistant":
            parts = self._assistant_parts(message)
            return {"role": "model", "parts": parts}

        if message.role == "tool":
            return {
                "role": "user",
                "parts": [self._tool_response_part(message)],
            }

        return {
            "role": "user",
            "parts": self._content_to_parts(message.content),
        }

    def _tool_response_part(self, message: UnifiedMessage) -> dict[str, Any]:
        return {
            "functionResponse": {
                "name": message.name or message.tool_call_id or "tool",
                "response": self._tool_response_payload(message.content),
            }
        }

    def _assistant_parts(self, message: UnifiedMessage) -> list[dict[str, Any]]:
        parts = self._content_to_parts(message.content) if message.content else []
        for call in message.tool_calls or []:
            parts.append(self._assistant_tool_call_part(call))
        return parts or [{"text": ""}]

    def _assistant_tool_call_part(self, call: dict[str, Any]) -> dict[str, Any]:
        if "functionCall" in call:
            function_call = call.get("functionCall") or {}
            part = {
                "functionCall": {
                    "name": str(function_call.get("name") or ""),
                    "args": self._coerce_arguments(function_call.get("args") or {}),
                }
            }
            self._copy_thought_fields(call, part)
            return part

        function = call.get("function") or {}
        name = function.get("name") or call.get("name")
        arguments = function.get(
            "arguments",
            call.get("arguments", call.get("args", {})),
        )
        part = {
            "functionCall": {
                "name": str(name or ""),
                "args": self._coerce_arguments(arguments),
            }
        }
        self._copy_thought_fields(call, part)
        return part

    def _copy_thought_fields(
        self,
        source: dict[str, Any],
        target: dict[str, Any],
    ) -> None:
        for key in ("thought", "thoughtSignature", "thought_signature"):
            if key in source:
                target[key] = source[key]

    def _content_to_parts(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"text": content}]

        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append({"text": str(item)})
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    parts.append({"text": str(item.get("text", ""))})
                    continue
                if item_type == "image_url":
                    image_part = self._image_url_part(item)
                    if image_part:
                        parts.append(image_part)
                    continue

                parts.append({"text": json.dumps(item, ensure_ascii=False)})
            return parts or [{"text": ""}]

        return [{"text": json.dumps(content, ensure_ascii=False)}]

    def _image_url_part(self, item: dict[str, Any]) -> dict[str, Any] | None:
        image_url = item.get("image_url") or {}
        if isinstance(image_url, str):
            url = image_url
            image_data: dict[str, Any] = {}
        elif isinstance(image_url, dict):
            url = image_url.get("url")
            image_data = image_url
        else:
            return None

        if not url:
            return None

        mime_type = self._image_mime_type(item, image_data, str(url))
        if str(url).startswith("data:"):
            inline_part = self._data_url_part(str(url), mime_type)
            return inline_part

        file_data = {"fileUri": str(url)}
        if mime_type:
            file_data["mimeType"] = mime_type
        return {"fileData": file_data}

    def _image_mime_type(
        self,
        item: dict[str, Any],
        image_data: dict[str, Any],
        url: str,
    ) -> str | None:
        for source in (image_data, item):
            for key in ("mimeType", "mime_type", "mime"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        if url.startswith("data:"):
            header = url.partition(",")[0]
            mime_type = header.removeprefix("data:").split(";", 1)[0].strip()
            return mime_type or None

        parsed = urlparse(url)
        guess_source = parsed.path or url
        mime_type, _ = mimetypes.guess_type(guess_source)
        return mime_type or "image/jpeg"

    def _data_url_part(
        self,
        url: str,
        mime_type: str | None,
    ) -> dict[str, Any] | None:
        header, separator, data = url.partition(",")
        if separator != ",":
            return None

        if not mime_type:
            mime_type = header.removeprefix("data:").split(";", 1)[0].strip()
        if not mime_type:
            return None

        return {"inlineData": {"mimeType": mime_type, "data": data}}

    def _tool_to_function_declaration(self, tool: dict[str, Any]) -> dict[str, Any]:
        declaration: dict[str, Any] = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
        }
        parameters = tool.get("parameters")
        if parameters:
            declaration["parameters"] = parameters
        return declaration

    def _tool_response_payload(self, content: Any) -> dict[str, Any]:
        if not isinstance(content, str):
            return {"content": content}

        try:
            return {"content": json.loads(content)}
        except json.JSONDecodeError:
            return {"content": content}

    def _coerce_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return {"_raw": arguments}
            return parsed if isinstance(parsed, dict) else {"_raw": parsed}
        return {"_raw": arguments}

    def _post_stream_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if session_id:
            headers["x-gateway-session-id"] = session_id

        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method="POST",
        )


        return _request_with_retries(
            req,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            reader=self._read_stream_objects,
        )

    def _read_stream_objects(self, resp: Any) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if not line or line == "[DONE]":
                continue

            parsed = json.loads(line)
            if isinstance(parsed, list):
                objects.extend(item for item in parsed if isinstance(item, dict))
            elif isinstance(parsed, dict):
                objects.append(parsed)
        return objects

    def _parse_stream_response(self, chunks: list[dict[str, Any]]) -> UnifiedResponse:
        text_parts: list[str] = []
        tool_calls: list[UnifiedToolCall] = []
        usage = None
        finish_reason = None

        for chunk in chunks:
            usage = chunk.get("usageMetadata") or usage
            for candidate in chunk.get("candidates") or []:
                finish_reason = candidate.get("finishReason") or finish_reason
                content = candidate.get("content") or {}
                for part in content.get("parts") or []:
                    if "text" in part:
                        text_parts.append(str(part["text"]))
                    function_call = part.get("functionCall")
                    if function_call:
                        raw_part = {"functionCall": function_call}
                        self._copy_thought_fields(part, raw_part)
                        tool_calls.append(
                            UnifiedToolCall(
                                id=f"tool_call_{len(tool_calls)}",
                                name=str(function_call.get("name") or ""),
                                arguments=self._coerce_arguments(
                                    function_call.get("args") or {}
                                ),
                                raw=raw_part,
                            )
                        )

        return UnifiedResponse(
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            raw={"chunks": chunks},
            usage=usage,
            finish_reason=self._normalize_finish_reason(finish_reason, tool_calls),
        )

    def _normalize_finish_reason(
        self,
        finish_reason: str | None,
        tool_calls: list[UnifiedToolCall],
    ) -> str | None:
        if tool_calls:
            return "tool_calls"
        if finish_reason is None:
            return None
        if finish_reason == "STOP":
            return "stop"
        return finish_reason.lower()




_RETRIABLE_STATUS = frozenset({429, 500, 502, 503, 504})







_RETRIABLE_CONN_ERRORS = (
    error.URLError,
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    ConnectionError,
    TimeoutError,
    socket.timeout,
)


def _request_with_retries(
    req: request.Request,
    *,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    reader: Callable[[Any], Any],
) -> Any:
    

    attempt = 0
    while True:
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return reader(resp)
        except error.HTTPError as exc:




            detail = exc.read().decode("utf-8", errors="replace")




            retriable = (
                exc.code in _RETRIABLE_STATUS or _body_signals_rate_limit(detail)
            )
            if not retriable or attempt >= max_retries:
                raise RuntimeError(f"provider HTTP {exc.code}: {detail}") from exc
            delay = _retry_delay(exc, attempt, retry_backoff)
            if delay > 0:
                time.sleep(delay)
            attempt += 1
        except _RETRIABLE_CONN_ERRORS:
            if attempt >= max_retries:
                raise
            delay = retry_backoff * (2**attempt)
            if delay > 0:
                time.sleep(delay)
            attempt += 1






_RATE_LIMIT_BODY_MARKERS = (
    "mpe-429",
    "resource_exhausted",
    "限流",
    "too many requests",
    "rate limit",
)


def _body_signals_rate_limit(detail: str) -> bool:
    if not detail:
        return False
    text = detail.lower()
    return any(marker in text for marker in _RATE_LIMIT_BODY_MARKERS)


def _retry_delay(
    exc: error.HTTPError,
    attempt: int,
    retry_backoff: float,
) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return retry_backoff * (2**attempt)


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

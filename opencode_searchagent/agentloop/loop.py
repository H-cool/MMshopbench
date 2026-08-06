from __future__ import annotations

import json
import re
import time
import uuid
from copy import deepcopy
from typing import Any, Callable

from .providers import ProviderAdapter
from .recorders import Recorder
from .tools import LocalTool, LocalToolExecutor, ToolDispatcher, ToolExecutor
from .types import AgentResult, UnifiedMessage, UnifiedRequest


class AgentLoop:
    def __init__(
        self,
        *,
        provider: ProviderAdapter,
        tools: list[LocalTool] | None = None,
        tool_dispatcher: ToolDispatcher | None = None,
        tool_executor: ToolExecutor | None = None,
        recorder: Recorder | None = None,

        max_steps: int = 16,
        sampling: dict[str, Any] | None = None,
        max_empty_retries: int = 3,
        response_validator: Callable[
            [str, list[UnifiedMessage]], str | None
        ]
        | None = None,
        max_validation_retries: int = 1,
    ) -> None:
        self.provider = provider
        self.tool_executor = tool_executor or LocalToolExecutor(
            tools or [],
            tool_dispatcher or _missing_tool_dispatcher,
        )
        self.recorder = recorder
        self.max_steps = max_steps
        self.sampling = sampling or {}
        self.max_empty_retries = max(0, max_empty_retries)






        self.response_validator = response_validator
        self.max_validation_retries = max(0, max_validation_retries)

    def run(
        self,
        messages: list[UnifiedMessage],
        *,
        run_id: str | None = None,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentResult:
        run_id = run_id or str(uuid.uuid4())
        started = time.perf_counter()
        input_messages = deepcopy(messages)
        current_messages = deepcopy(messages)
        steps: list[dict[str, Any]] = []
        final_text = ""
        finish_reason = "max_steps"
        error_message = None
        validation_retries = 0

        try:
            for step_index in range(self.max_steps):
                unified_request = UnifiedRequest(
                    messages=deepcopy(current_messages),
                    tools=self._tool_schemas(),
                    model=getattr(self.provider, "model", None),
                    sampling=self.sampling,
                    session_id=run_id,
                )

                response = self._complete_with_retry(unified_request)

                output_message = UnifiedMessage(
                    role="assistant",
                    content=response.text or "",
                    tool_calls=[
                        tool_call.to_dict() for tool_call in response.tool_calls
                    ]
                    or None,
                )
                step_record: dict[str, Any] = {
                    "index": step_index,
                    "input": {
                        "messages": [
                            _message_to_log_dict(message)
                            for message in unified_request.messages
                        ],
                    },
                    "output": {
                        "messages": [_message_to_log_dict(output_message)],
                    },
                }
                response_metadata = _response_metadata_from_raw(response.raw)
                if response_metadata:
                    step_record["response_metadata"] = response_metadata

                if response.tool_calls:
                    current_messages.append(
                        UnifiedMessage(
                            role="assistant",
                            content=response.text or "",
                            tool_calls=[
                                _tool_call_to_runtime_dict(tool_call.to_dict())
                                for tool_call in response.tool_calls
                            ],
                        )
                    )









                    follow_up_images: list[dict[str, Any]] = []
                    seen_urls: set[str] = set()
                    for tool_call in response.tool_calls:
                        tool_result = self.tool_executor.execute(
                            tool_call,
                            run_id=run_id,
                            step_index=step_index,
                            messages=deepcopy(current_messages),
                        )
                        current_messages.append(tool_result.to_message())
                        for entry in (tool_result.metadata or {}).get(
                            "follow_up_images", []
                        ):



                            if isinstance(entry, str):
                                url = entry
                                meta: dict[str, Any] = {"url": entry}
                            elif isinstance(entry, dict):
                                url = entry.get("url")
                                meta = dict(entry)
                            else:
                                continue





                            if (
                                isinstance(url, str)
                                and url.startswith(("http://", "https://"))
                                and not _WHITESPACE_OR_ESCAPE_RE.search(url)
                                and url not in seen_urls
                            ):
                                seen_urls.add(url)
                                follow_up_images.append(meta)
                    if follow_up_images:
                        current_messages.append(
                            _tool_images_user_message(follow_up_images)
                        )
                    steps.append(step_record)
                    continue

                final_text = response.text or ""
                finish_reason = response.finish_reason or "final_answer"




                if (
                    self.response_validator is not None
                    and validation_retries < self.max_validation_retries
                ):
                    correction = self.response_validator(
                        final_text, deepcopy(current_messages)
                    )
                    if correction:
                        validation_retries += 1
                        current_messages.append(
                            UnifiedMessage(role="assistant", content=final_text)
                        )
                        current_messages.append(
                            UnifiedMessage(role="user", content=correction)
                        )
                        steps.append(step_record)
                        continue

                current_messages.append(
                    UnifiedMessage(role="assistant", content=final_text)
                )
                steps.append(step_record)
                break
        except Exception as exc:
            error_message = str(exc)
            finish_reason = "model_error"

        result = AgentResult(
            final_text=final_text,
            finish_reason=finish_reason,
            messages=current_messages,
            steps=steps,
            run_id=run_id,
            error=error_message,
        )
        self._record_run(
            result=result,
            input_messages=input_messages,
            task_id=task_id,
            metadata=metadata or {},
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return result

    def _complete_with_retry(self, unified_request: UnifiedRequest):
        
        response = self.provider.complete(unified_request)
        attempts = 0
        while (
            attempts < self.max_empty_retries
            and not response.tool_calls
            and not (response.text or "").strip()
        ):
            attempts += 1
            response = self.provider.complete(unified_request)
        return response

    def _tool_schemas(self) -> list[dict[str, Any]]:
        schemas = getattr(self.tool_executor, "schemas", None)
        if callable(schemas):
            return schemas()
        return []

    def _record_run(
        self,
        *,
        result: AgentResult,
        input_messages: list[UnifiedMessage],
        task_id: str | None,
        metadata: dict[str, Any],
        latency_ms: float,
    ) -> None:
        if self.recorder is None:
            return

        self.recorder.record(
            {
                "run_id": result.run_id,
                "task_id": task_id,
                "metadata": metadata,
                "provider": type(self.provider).__name__,
                "model": getattr(self.provider, "model", None),
                "sampling": self.sampling,
                "tools": [_tool_to_log_dict(tool) for tool in self._tool_schemas()],
                "steps": result.steps,
                "final": {
                    "text": result.final_text,
                    "finish_reason": result.finish_reason,
                    "error": result.error,
                    "latency_ms": latency_ms,
                },
            }
        )





_WHITESPACE_OR_ESCAPE_RE = re.compile(r"\s|\\[nrtf]")


def _tool_images_user_message(images: list[dict[str, Any]]) -> UnifiedMessage:
    
    _type_cn = {"main": "主图", "sku": "规格图", "detail": "详情图", "gallery": "相册图"}
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"以下是上一步工具返回的商品图片（共 {len(images)} 张）。"
                "每张图片前都标注了它来自哪个商品 id 的哪类图片"
                "（main=主图 / sku=规格图 / detail=详情图），"
                "请对照标注逐张查看后再继续下一步。"
            ),
        }
    ]
    for img in images:
        if isinstance(img, dict):
            url = img.get("url")
            item_id = img.get("item_id")
            pic_type = img.get("type")
            pos = img.get("pos")
        else:
            url = img
            item_id = pic_type = pos = None
        if not isinstance(url, str) or not url:
            continue
        label_bits: list[str] = []
        if item_id:
            label_bits.append(f"item_id={item_id}")
        if pic_type:
            cn = _type_cn.get(str(pic_type), str(pic_type))
            label_bits.append(f"type={pic_type}({cn})")
        if isinstance(pos, int):
            label_bits.append(f"第{pos + 1}张")
        label = " ".join(label_bits) if label_bits else "商品图片"
        content.append({"type": "text", "text": f"[{label}]"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return UnifiedMessage(role="user", content=content)


def _message_to_log_dict(message: UnifiedMessage) -> dict[str, Any]:
    data: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.name is not None:
        data["name"] = message.name
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    if message.tool_calls is not None:
        data["tool_calls"] = [
            _tool_call_to_log_dict(tool_call, index)
            for index, tool_call in enumerate(message.tool_calls)
        ]
    return data


def _tool_call_to_runtime_dict(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call.get("raw")
    if isinstance(raw, dict):
        data = deepcopy(raw)
        data.setdefault("id", tool_call.get("id"))
        return data
    return tool_call


def _tool_call_to_log_dict(tool_call: dict[str, Any], index: int) -> dict[str, Any]:
    if "type" in tool_call and "function" in tool_call:
        data = deepcopy(tool_call)
        function = data.get("function") or {}
        arguments = function.get("arguments", "{}")
        function["arguments"] = _arguments_to_json_string(arguments)
        data["function"] = function
        return data

    raw = tool_call.get("raw")
    if isinstance(raw, dict) and "type" in raw and "function" in raw:
        data = deepcopy(raw)
        function = data.get("function") or {}
        arguments = function.get("arguments", "{}")
        function["arguments"] = _arguments_to_json_string(arguments)
        data["function"] = function
        if "id" not in data and tool_call.get("id") is not None:
            data["id"] = str(tool_call["id"])
        return data

    function_call = tool_call.get("functionCall")
    if not isinstance(function_call, dict) and isinstance(raw, dict):
        function_call = raw.get("functionCall")

    if isinstance(function_call, dict):
        name = function_call.get("name") or tool_call.get("name") or ""
        arguments = function_call.get("args", tool_call.get("arguments", {}))
    else:
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name") or ""
        arguments = function.get(
            "arguments",
            tool_call.get("arguments", tool_call.get("args", {})),
        )

    return {
        "id": str(tool_call.get("id") or f"tool_call_{index}"),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": _arguments_to_json_string(arguments),
        },
    }


def _tool_to_log_dict(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return deepcopy(tool)
    return {
        "type": "function",
        "function": deepcopy(tool),
    }


def _arguments_to_json_string(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def _response_metadata_from_raw(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    extend_fields = raw.get("extend_fields")
    if not isinstance(extend_fields, dict):
        return {}

    metadata = {}
    for key in ("requestId", "traceId"):
        value = extend_fields.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _missing_tool_dispatcher(_: Any) -> Any:
    raise RuntimeError("tool_dispatcher is required when tools are configured")

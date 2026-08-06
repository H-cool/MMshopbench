from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .types import ToolResult, UnifiedMessage, UnifiedToolCall, to_jsonable


@dataclass
class ToolExecutionContext:
    run_id: str
    step_index: int
    tool_call_id: str
    messages: list[UnifiedMessage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_index": self.step_index,
            "tool_call_id": self.tool_call_id,
            "messages": [message.to_dict() for message in self.messages],
        }


@dataclass
class LocalToolRequest:
    name: str
    input: Any
    context: ToolExecutionContext

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "input": to_jsonable(self.input),
            "context": self.context.to_dict(),
        }


ToolConverter = Callable[[UnifiedToolCall, ToolExecutionContext], LocalToolRequest]
ToolDispatcher = Callable[[LocalToolRequest], Any]
ToolArgumentConverter = Callable[
    [UnifiedToolCall, ToolExecutionContext],
    dict[str, Any],
]


@dataclass
class LocalTool:
    name: str
    description: str
    parameters: dict[str, Any]
    convert: ToolConverter

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    convert: ToolArgumentConverter | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolExecutor(Protocol):
    def schemas(self) -> list[dict[str, Any]]:
        ...

    def execute(
        self,
        tool_call: UnifiedToolCall,
        *,
        run_id: str,
        step_index: int,
        messages: list[UnifiedMessage],
    ) -> ToolResult:
        ...


class LocalToolExecutor:
    def __init__(
        self,
        tools: list[LocalTool],
        dispatcher: ToolDispatcher,
    ) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.dispatcher = dispatcher

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def execute(
        self,
        tool_call: UnifiedToolCall,
        *,
        run_id: str,
        step_index: int,
        messages: list[UnifiedMessage],
    ) -> ToolResult:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                ok=False,
                content=f"Tool not found: {tool_call.name}",
                error="tool_not_found",
            )

        context = ToolExecutionContext(
            run_id=run_id,
            step_index=step_index,
            tool_call_id=tool_call.id,
            messages=messages,
        )
        start = time.perf_counter()

        try:
            local_request = tool.convert(tool_call, context)
            result = self.dispatcher(local_request)
            latency_ms = (time.perf_counter() - start) * 1000
            return _to_tool_result(
                result,
                tool_call=tool_call,
                tool_name=local_request.name,
                request=local_request.to_dict(),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool.name,
                ok=False,
                content=f"Tool error: {exc}",
                error=str(exc),
                latency_ms=latency_ms,
            )


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
        convert: ToolArgumentConverter | None = None,
    ) -> RegisteredTool:
        tool = RegisteredTool(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
            convert=convert,
        )
        self.tools[name] = tool
        return tool

    def tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        convert: ToolArgumentConverter | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                name=name,
                description=description,
                parameters=parameters,
                func=func,
                convert=convert,
            )
            return func

        return decorator

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def execute(
        self,
        tool_call: UnifiedToolCall,
        *,
        run_id: str,
        step_index: int,
        messages: list[UnifiedMessage],
    ) -> ToolResult:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                ok=False,
                content=f"Tool not found: {tool_call.name}",
                error="tool_not_found",
            )

        context = ToolExecutionContext(
            run_id=run_id,
            step_index=step_index,
            tool_call_id=tool_call.id,
            messages=messages,
        )
        start = time.perf_counter()

        try:
            kwargs = tool.convert(tool_call, context) if tool.convert else tool_call.arguments
            result = tool.func(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            return _to_tool_result(
                result,
                tool_call=tool_call,
                tool_name=tool.name,
                request={
                    "name": tool.name,
                    "arguments": to_jsonable(kwargs),
                    "context": context.to_dict(),
                },
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool.name,
                ok=False,
                content=f"Tool error: {exc}",
                error=str(exc),
                latency_ms=latency_ms,
            )


def _to_tool_result(
    value: Any,
    *,
    tool_call: UnifiedToolCall,
    tool_name: str,
    request: dict[str, Any],
    latency_ms: float,
) -> ToolResult:
    if isinstance(value, ToolResult):
        if value.latency_ms is None:
            value.latency_ms = latency_ms
        if value.request is None:
            value.request = request
        return value

    if isinstance(value, str):
        content = value
    else:
        content = json.dumps(to_jsonable(value), ensure_ascii=False)

    return ToolResult(
        tool_call_id=tool_call.id,
        name=tool_name,
        ok=True,
        content=content,
        request=request,
        latency_ms=latency_ms,
    )

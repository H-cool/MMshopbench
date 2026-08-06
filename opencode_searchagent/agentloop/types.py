from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]
MessageContent = str | list[dict[str, Any]]


@dataclass
class UnifiedMessage:
    role: Role
    content: MessageContent
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            data["name"] = self.name
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        return data


@dataclass
class UnifiedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }
        if self.raw is not None:
            data["raw"] = self.raw
        return data


@dataclass
class UnifiedRequest:
    messages: list[UnifiedMessage]
    tools: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None
    sampling: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "messages": [message.to_dict() for message in self.messages],
            "tools": self.tools,
            "model": self.model,
            "sampling": self.sampling,
        }
        if self.session_id is not None:
            data["session_id"] = self.session_id
        return data


@dataclass
class UnifiedResponse:
    text: str | None = None
    tool_calls: list[UnifiedToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "raw": self.raw,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
        }


@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    latency_ms: float | None = None

    def to_message(self) -> UnifiedMessage:
        return UnifiedMessage(
            role="tool",
            content=self.content,
            name=self.name,
            tool_call_id=self.tool_call_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "ok": self.ok,
            "content": self.content,
            "metadata": self.metadata,
            "error": self.error,
            "request": self.request,
            "response": self.response,
            "latency_ms": self.latency_ms,
        }


@dataclass
class AgentResult:
    final_text: str
    finish_reason: str
    messages: list[UnifiedMessage]
    steps: list[dict[str, Any]]
    run_id: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_text": self.final_text,
            "finish_reason": self.finish_reason,
            "messages": [message.to_dict() for message in self.messages],
            "steps": self.steps,
            "run_id": self.run_id,
            "error": self.error,
        }


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value

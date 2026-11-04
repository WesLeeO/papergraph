"""Minimal chat client for a local model served by Ollama (https://ollama.com).

Uses Ollama's native /api/chat, which supports tool calling and returns a
model's reasoning ("thinking") separately from its answer. One small class on
purpose: the agent loop only needs `chat(messages, tools) -> Reply`, so another
backend (e.g. Claude) can be swapped in by writing the same method.
"""

from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://localhost:11434"


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Reply:
    content: str
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict[str, Any]:
        """The assistant turn to append to the conversation history."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls
            ]
        return msg


class LLMError(RuntimeError):
    pass


# structural typing related to ChatModel
class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        think: bool = False,
        num_ctx: int = 16384,
        timeout_s: float = 300,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.think = think
        # Ollama's default context is small (a few thousand tokens) and it
        # silently drops the oldest messages beyond it; an agent's history of
        # tool results needs more.
        self.num_ctx = num_ctx
        self.http = httpx.Client(timeout=timeout_s)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        if tools:
            body["tools"] = tools
        try:
            resp = self.http.post(f"{self.host}/api/chat", json=body)
            """
            {
              "model": "llama3",
              "created_at": "2026-11-04T10:40:00.123456Z",
              "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
                "images": null,
                "tool_calls": null
              },
              "done_reason": "stop",
              "done": true,
              "total_duration": 456789000,
              "load_duration": 1234000,
              "prompt_eval_count": 12,
              "prompt_eval_duration": 456000,
              "eval_count": 8,
              "eval_duration": 40234000
            }
            """
        except httpx.HTTPError as e:
            raise LLMError(
                f"cannot reach Ollama at {self.host} ({e}); is `ollama serve` running?"
            ) from e
        if resp.status_code != 200:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        msg = resp.json().get("message", {})
        calls = [
            ToolCall(c["function"]["name"], c["function"].get("arguments") or {})
            for c in msg.get("tool_calls") or []
        ]

        # content + tools to be called
        return Reply(
            content=msg.get("content", ""), thinking=msg.get("thinking", ""), tool_calls=calls
        )

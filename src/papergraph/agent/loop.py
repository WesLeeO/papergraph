"""The agent loop: ask the model, run the tools it calls, repeat, then answer.

    messages = [system prompt, question]
    repeat up to max_steps:
        reply = model(messages, tools)
        no tool calls?  -> that's the final answer
        else            -> run each tool, append its result, loop

The loop itself is short; the work is in the guardrails a small local model
needs: a step limit (then a forced answer), repeated-call detection, tool
errors fed back as text so the model can correct itself, and citation checking
(only papers actually seen in tool results count).

`run_agent` yields events as it goes, so the web UI can stream each step live.
"""

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from papergraph.agent.llm import LLMError, Reply
from papergraph.agent.tools import ToolResult
from papergraph.search import Paper

PROMPT_VERSION = "agent_system_v1"
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
MAX_STEPS = 6
MIN_SEARCHES = 2  # small models tend to stop after one search; nudge them once
NUDGE = (
    "Before answering: you have run only one search. Questions usually have "
    "several aspects. Search for the aspects your results do not cover yet "
    "(use different, specific queries), then answer."
)


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Reply: ...


class Tools(Protocol):
    def specs(self) -> list[dict[str, Any]]: ...

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...


@dataclass
class Event:
    type: str  # thinking | tool_call | tool_result | nudge | final | error
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"type": self.type, **self.data})


def system_prompt(version: str = PROMPT_VERSION) -> str:
    return (PROMPTS_DIR / f"{version}.md").read_text()


_CITATION = re.compile(r"\[([A-Za-z0-9.\-/]+)\]")


def cited_ids(answer: str) -> list[str]:
    """Paper ids cited as [id] in the answer, in order, without repeats."""
    return list(dict.fromkeys(_CITATION.findall(answer)))


def call_key(name: str, arguments: dict[str, Any]) -> str:
    return name + json.dumps(arguments, sort_keys=True, default=str).lower()


def run_agent(
    question: str,
    model: ChatModel,
    tools: Tools,
    max_steps: int = MAX_STEPS,
    min_searches: int = MIN_SEARCHES,
) -> Iterator[Event]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": question},
    ]
    seen_papers: dict[str, Paper] = {}
    seen_calls: set[str] = set()
    n_calls = 0
    nudged = False
    start = time.perf_counter()

    def final(reply: Reply, steps: int) -> Event:
        ids = cited_ids(reply.content)
        return Event(
            "final",
            {
                "answer": reply.content,
                "papers": [seen_papers[i].to_dict() for i in ids if i in seen_papers],
                "unverified_ids": [i for i in ids if i not in seen_papers],
                "steps": steps,
                "seconds": round(time.perf_counter() - start, 1),
                "prompt_version": PROMPT_VERSION,
            },
        )

    for step in range(1, max_steps + 1):
        try:
            reply = model.chat(messages, tools.specs())
        except LLMError as e:
            yield Event("error", {"message": str(e)})
            return
        if reply.thinking:
            yield Event("thinking", {"step": step, "text": reply.thinking})
        messages.append(reply.as_message())
        if not reply.tool_calls:
            if n_calls < min_searches and not nudged and step < max_steps:
                nudged = True  # once: if it still answers, accept the answer
                yield Event("nudge", {"step": step, "message": NUDGE})
                messages.append({"role": "user", "content": NUDGE})
                continue
            yield final(reply, step - 1)
            return
        for call in reply.tool_calls:
            n_calls += 1
            yield Event("tool_call", {"step": step, "name": call.name, "arguments": call.arguments})
            key = call_key(call.name, call.arguments)
            if key in seen_calls:
                result = ToolResult(
                    "You already made this exact call; its results are above. "
                    "Use different arguments or write your final answer."
                )
            else:
                seen_calls.add(key)
                result = tools.run(call.name, call.arguments)
            for p in result.papers:
                seen_papers[p.id] = p
            yield Event(
                "tool_result",
                {
                    "step": step,
                    "name": call.name,
                    "n_papers": len(result.papers),
                    "papers": [p.to_dict() for p in result.papers],
                    "error": result.text
                    if result.text.startswith(("Error", "You already"))
                    else None,
                },
            )
            messages.append({"role": "tool", "tool_name": call.name, "content": result.text})

    # Out of steps: ask for the answer with tools withheld.
    messages.append(
        {
            "role": "user",
            "content": "Step limit reached. Write your final answer now, citing only "
            "papers from the tool results above.",
        }
    )
    try:
        reply = model.chat(messages, None)
    except LLMError as e:
        yield Event("error", {"message": str(e)})
        return
    yield final(reply, max_steps)

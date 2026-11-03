"""Agent loop and tools, with a scripted fake model: no Ollama or database needed."""

from typing import Any

from papergraph.agent.llm import LLMError, Reply, ToolCall
from papergraph.agent.loop import Event, cited_ids, run_agent
from papergraph.agent.tools import (
    Toolbox,
    ToolResult,
    arxiv_query,
    as_list,
    clamp,
    parse_arxiv_feed,
)
from papergraph.search import Paper


def paper(pid: str) -> Paper:
    return Paper(
        pid, f"Title {pid}", "Abstract.", ["A. Author"], "2024-01-01", None, None, ["cs.CL"]
    )


class FakeModel:
    """Returns scripted replies in order; records what it was sent."""

    def __init__(self, replies: list[Reply | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[int, bool]] = []  # (n_messages, tools offered?)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Reply:
        self.calls.append((len(messages), tools is not None))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeTools:
    def __init__(self) -> None:
        self.runs: list[tuple[str, dict[str, Any]]] = []

    def specs(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "search_papers"}}]

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        self.runs.append((name, arguments))
        return ToolResult("[2401.00001] Title ...", [paper("2401.00001"), paper("2401.00002")])


def search_call(query: str = "rag hallucination") -> Reply:
    return Reply("", tool_calls=[ToolCall("search_papers", {"query": query})])


def events_of(evs: list[Event], kind: str) -> list[Event]:
    return [e for e in evs if e.type == kind]


def test_tool_call_then_final_answer() -> None:
    model = FakeModel([search_call(), Reply("Read [2401.00001] and [9999.99999].")])
    tools = FakeTools()
    evs = list(run_agent("q", model, tools, min_searches=1))
    assert [e.type for e in evs] == ["tool_call", "tool_result", "final"]
    final = evs[-1].data
    # Only papers seen in tool results count; invented ids are flagged.
    assert [p["id"] for p in final["papers"]] == ["2401.00001"]
    assert final["unverified_ids"] == ["9999.99999"]
    assert final["steps"] == 1
    assert tools.runs == [("search_papers", {"query": "rag hallucination"})]


def test_answering_after_one_search_gets_one_nudge() -> None:
    replies = [search_call("a"), Reply("too early"), search_call("b"), Reply("ok [2401.00001]")]
    model = FakeModel(replies)
    tools = FakeTools()
    evs = list(run_agent("q", model, tools))
    assert [e.type for e in evs].count("nudge") == 1
    assert len(tools.runs) == 2
    assert evs[-1].type == "final" and evs[-1].data["answer"] == "ok [2401.00001]"


def test_nudge_happens_only_once() -> None:
    model = FakeModel([search_call("a"), Reply("early"), Reply("still early")])
    evs = list(run_agent("q", model, FakeTools()))
    assert [e.type for e in evs].count("nudge") == 1
    assert evs[-1].data["answer"] == "still early"  # accepted the second time


def test_repeated_call_is_not_rerun() -> None:
    model = FakeModel([search_call(), search_call(), Reply("done")])
    tools = FakeTools()
    evs = list(run_agent("q", model, tools))
    assert len(tools.runs) == 1
    second = events_of(evs, "tool_result")[1].data
    assert second["error"].startswith("You already")


def test_step_limit_forces_an_answer_without_tools() -> None:
    model = FakeModel([search_call(f"q{i}") for i in range(3)] + [Reply("final [2401.00002]")])
    evs = list(run_agent("q", model, FakeTools(), max_steps=3))
    assert evs[-1].type == "final"
    assert evs[-1].data["steps"] == 3
    assert model.calls[-1][1] is False  # tools withheld on the forced last call
    assert [p["id"] for p in evs[-1].data["papers"]] == ["2401.00002"]


def test_llm_failure_becomes_an_error_event() -> None:
    evs = list(run_agent("q", FakeModel([LLMError("Ollama down")]), FakeTools()))
    assert [e.type for e in evs] == ["error"]
    assert "Ollama down" in evs[0].data["message"]


def test_cited_ids() -> None:
    text = "See [2310.06825], [cs/0112017] and again [2310.06825]; not a [link text](x)."
    assert cited_ids(text) == ["2310.06825", "cs/0112017"]


# --- tools ---------------------------------------------------------------------------


def toolbox() -> Toolbox:
    return Toolbox(conn=None)  # type: ignore[arg-type]  # validation needs no database


def test_unknown_tool_and_missing_arguments_are_reported_to_the_model() -> None:
    tb = toolbox()
    assert tb.run("delete_everything", {}).text.startswith("Error: unknown tool")
    assert "needs argument(s) query" in tb.run("search_papers", {}).text


def test_tool_exceptions_become_text() -> None:
    # No database: the search raises; the model gets an error message instead of a crash.
    result = toolbox().run("search_papers", {"query": "x", "invented_arg": 1})
    assert result.text.startswith("Error running search_papers")


def test_specs_are_valid_function_schemas() -> None:
    specs = toolbox().specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {"search_papers", "get_paper", "similar_papers", "search_arxiv"}
    for s in specs:
        assert s["type"] == "function"
        assert s["function"]["parameters"]["type"] == "object"


def test_argument_coercion() -> None:
    assert clamp("12", 1, 15, 8) == 12
    assert clamp(99, 1, 15, 8) == 15
    assert clamp("lots", 1, 15, 8) == 8
    assert as_list("ICLR, NeurIPS") == ("ICLR", "NeurIPS")
    assert as_list(["ICML"]) == ("ICML",)
    assert as_list(None) == ()


def test_arxiv_query_uses_key_terms() -> None:
    assert arxiv_query("sparse attention for long documents") == (
        "all:sparse AND all:attention AND all:for AND all:long AND all:documents"
    )


ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2511.01234v2</id>
    <published>2026-11-01T10:00:00Z</published>
    <title>A  New
      Method</title>
    <summary>We propose   a method.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <category term="cs.LG"/>
  </entry>
</feed>"""


def test_parse_arxiv_feed() -> None:
    (p,) = parse_arxiv_feed(ATOM)
    assert p.id == "2511.01234"  # version suffix stripped
    assert p.title == "A New Method"
    assert p.abstract == "We propose a method."
    assert p.authors == ["Ada Lovelace", "Alan Turing"]
    assert p.published == "2026-11-01"
    assert p.categories == ["cs.LG"]
    assert p.url == "https://arxiv.org/abs/2511.01234"

"""Run the research agent from the terminal:

uv run --group embed python -m papergraph.agent "your research question" [--schema public]
"""

import argparse
import json

from papergraph.agent.llm import DEFAULT_MODEL, OllamaClient
from papergraph.agent.loop import run_agent
from papergraph.agent.tools import Toolbox
from papergraph.db import connect


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("question")
    parser.add_argument("--schema", default="public", help="corpus to search")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--think", action="store_true", help="let the model reason before acting")
    args = parser.parse_args()

    with connect(args.schema, autocommit=True) as conn:
        for event in run_agent(
            args.question, OllamaClient(args.model, think=args.think), Toolbox(conn)
        ):
            d = event.data
            if event.type == "tool_call":
                print(f"\n[step {d['step']}] {d['name']}({json.dumps(d['arguments'])})")
            elif event.type == "tool_result":
                print(f"   -> {d['error'] or str(d['n_papers']) + ' papers'}")
            elif event.type == "nudge":
                print("   (guardrail: asked the agent to search the other aspects first)")
            elif event.type == "thinking":
                print(f"   (thinking: {d['text'][:200]}...)")
            elif event.type == "final":
                print(
                    f"\n=== answer ({d['steps']} tool steps, {d['seconds']} s) ===\n{d['answer']}"
                )
                if d["unverified_ids"]:
                    print(f"\nWARNING: cited ids not seen in tool results: {d['unverified_ids']}")
            elif event.type == "error":
                print(f"ERROR: {d['message']}")


if __name__ == "__main__":
    main()

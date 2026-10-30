"""Problem 11 (compaction): keep every model call within config.context_budget.

The context is the list of messages after the system prompt. When the next call
would not fit, compact() shortens the context in steps, cheapest first, and stops
as soon as the call fits:
1. Old tool results become a short stub. A terminal result keeps the path of its
   saved output, so the model can read it again.
2. Past turns keep only the user's messages and the final answers.
3. Past turns are replaced by a summary that aux_lm writes, which keeps every
   requirement and constraint the user stated.
4. The current turn drops its oldest steps.
Never compacted away: the system prompt, the current user message, and the newest
reply with its tool result, which must reach the model verbatim. A long tool result is
also cut when it arrives (shorten), so one result can never fill the budget alone.
"""

from __future__ import annotations

import re

from cs329z_hw1.agent.tool_protocol import format_tool_result
from cs329z_hw1.tokens import CHARS_PER_TOKEN, count_message_tokens, count_tokens
from cs329z_hw1.types import ToolResult

RESULT_SHARE = 4  # one tool result may use at most 1/RESULT_SHARE of the budget
SUMMARY_SHARE = 8  # the summary may use at most 1/SUMMARY_SHARE of the budget
SUMMARY_HEADING = "## Summary of the earlier conversation"
SUMMARY_PROMPT = (
    "Summarize the earlier part of a conversation between a user and an assistant, so that "
    "the assistant can keep handling the user's later requests without it. The conversation "
    "may start with an earlier summary: fold it into the new one. Keep every requirement, "
    "constraint and preference the user stated, with its exact numbers, names and dates, and "
    "the main facts and answers found so far. Use at most {words} words, and reply with the "
    "summary only."
)


def shorten(text: str, max_tokens: int) -> str:
    """text if it fits in max_tokens; else its start and end with a note in between."""
    if count_tokens(text) <= max_tokens:
        return text
    half = (max_tokens * CHARS_PER_TOKEN - 200) // 2  # 200 characters are left for the note
    cut = len(text) - 2 * half
    return (f"{text[:half]}\n[... {cut} characters cut to fit the context. Ask for less, for example "
            f"a smaller k or a narrower command.]\n{text[-half:]}")


def payload(system: str, summary: str, context: list[dict]) -> list[dict]:
    """The messages of one model call. The summary, if any, ends the system prompt."""
    head = f"{system}\n\n{SUMMARY_HEADING}\n{summary}" if summary else system
    return [{"role": "system", "content": head}] + [{"role": e["role"], "content": e["content"]} for e in context]


def compact(system: str, summary: str, context: list[dict], budget: int, aux_lm) -> str:
    """Shorten context in place until the next call fits budget, and return the
    summary to use. Each context entry is {"role", "content", "event"}, where
    event is its transcript event; final answers also have "answer": True."""
    def fits() -> bool:
        return count_message_tokens(payload(system, summary, context)) <= budget

    current = max(i for i, e in enumerate(context) if e["event"]["type"] == "user")

    # 1. Old tool results become stubs, oldest first. The newest entry stays as it is.
    for e in context[:-1]:
        if fits():
            return summary
        if e["event"]["type"] == "tool_result" and not e.get("stub"):
            e["content"], e["stub"] = _stub(e), True

    # 2. Past turns drop their tool calls, tool results and errors, oldest first.
    while not fits():
        i = next((i for i, e in enumerate(context[:current])
                  if e["event"]["type"] != "user" and not e.get("answer")), None)
        if i is None:
            break
        del context[i]
        current -= 1

    # 3. Past turns become a summary written by aux_lm.
    if not fits() and current > 0:
        summary = _summarize(summary, context[:current], aux_lm, budget // SUMMARY_SHARE)
        del context[:current]
        current = 0

    # 4. The current turn drops its oldest steps, keeping its user message and the
    # newest reply with its result (newest tool call + tool result).
    while not fits() and current + 1 < len(context) - 2:
        del context[current + 1]
    return summary if fits() else ""  # last resort: drop the summary too


def _stub(entry: dict) -> str:
    """A short tool_result message in place of an old one."""
    event = entry["event"]
    note = "[This old result was removed to save space."
    paths = re.findall(r"saved in (\S+);", event["content"])
    if paths:
        note += f" The whole output is still saved in {', '.join(paths)}."
    return format_tool_result(event["name"], ToolResult(event["status"], note + "]"))


def _summarize(summary: str, entries: list[dict], aux_lm, max_tokens: int) -> str:
    """One summary of the earlier summary and the given entries, from aux_lm."""
    names = {"user": "User", "assistant": "Assistant", "tool_result": "Tool result", "error": "Error"}
    lines = [f"Earlier summary: {summary}"] if summary else []
    lines += [f"{names[e['event']['type']]}: {e['content']}" for e in entries]
    max_chars = max_tokens * CHARS_PER_TOKEN
    # The prompt asks for words, since a model keeps to a word limit far better than to a
    # character limit. A word with its space is about 6 characters, so this leaves some room.
    words = max_chars // 8
    try:
        text = aux_lm([{"role": "system", "content": SUMMARY_PROMPT.format(words=words)},
                       {"role": "user", "content": "\n\n".join(lines)}])
    except Exception:
        # No summary from the model: keep what the user said, which holds their constraints.
        text = "\n".join(line for line in lines if line.startswith(("Earlier summary:", "User:")))
    return text.strip()[:max_chars]  # the hard limit, if the summary is still too long

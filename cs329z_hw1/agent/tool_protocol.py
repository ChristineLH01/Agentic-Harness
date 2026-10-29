"""Problem 6 (tool_protocol): the text format for tool calls.

A tool call is one JSON object {"name": ..., "args": {...}} between <tool_call> tags.
"""

from __future__ import annotations

import json

from cs329z_hw1.types import ParsedResponse, ToolCall, ToolResult

START, END = "<tool_call>", "</tool_call>"

PROTOCOL = f"""\
## How to call a tool

To call a tool, end your reply with a call block:

{START}
{{"name": "TOOL_NAME", "args": {{"ARGUMENT_NAME": VALUE}}}}
{END}

- Follow this format strictly, including both tags.
- The block holds exactly one JSON object with "name" and "args".
- Use only the argument names listed for the tool, and each argument value should be \
a JSON value of the type listed for it.
- You can make at most one call, so each reply can contain at most one call block.
- When you need no tool, reply in plain text without any tags."""


def render_tools(schemas: list[dict]) -> str:
    """System-prompt text that lists the tools and explains how to call them."""
    tools = "\n\n".join(_render_tool(s) for s in schemas) or "(no tools)"
    return f"## Tools\n\n{tools}\n\n{PROTOCOL}"


def _render_tool(schema: dict) -> str:
    """Render one tool schema as a heading, description, and one line per parameter."""
    lines = [f"### {schema['name']}", (schema.get("description") or "").strip()]
    params = schema.get("parameters") or {}
    required = set(params.get("required", []))

    # One line per tool parameter
    for name, spec in (params.get("properties") or {}).items():
        kind = spec.get("type", "any")
        facts = [" or ".join(kind) if isinstance(kind, list) else kind]
        facts.append("required" if name in required else "optional")
        # Any other schema keyword (minimum, maximum, enum, ...) is shown verbatim as JSON.
        facts += [f"{k} {json.dumps(v, ensure_ascii=False)}" for k, v in spec.items() if k not in ("type", "description")]
        lines.append(f"- {name} ({', '.join(facts)})" + (f": {spec['description']}" if spec.get("description") else ""))

    if not params.get("properties"):
        lines.append("Takes no arguments.")
    return "\n".join(line for line in lines if line)


def format_tool_call(text: str, call: ToolCall) -> str:
    """Write a reply with ``text`` followed by ``call`` in designed format."""
    payload = json.dumps({"name": call.name, "args": call.args}, ensure_ascii=False)
    return "\n\n".join(part for part in (text.strip(), f"{START}\n{payload}\n{END}") if part)


def format_tool_result(name: str, result: ToolResult) -> str:
    """How a tool result is written back to the model. Used by the agent loop (Problem 7)"""
    return f'<tool_result tool="{name}" status="{result.status}">\n{result.content}\n</tool_result>'


HOW_TO_FIX = f'Write exactly one call as {START}{{"name": ..., "args": {{...}}}}{END}, or answer in plain text.'
_DECODER = json.JSONDecoder()

def parse_response(reply: str) -> ParsedResponse:
    """Split a reply into its text and at most one tool call."""
    try:
        return _parse(str(reply))
    except Exception as exc:  # e.g. RecursionError on absurdly nested JSON
        return ParsedResponse(text=str(reply).strip(), error=f"The reply could not be read ({exc}). {HOW_TO_FIX}")


def _parse(reply: str) -> ParsedResponse:
    before, opened, after = reply.partition(START)
    # No opening tag: a plain-text answer.
    if not opened:
        return ParsedResponse(text=reply.strip())

    # Read exactly one JSON value after the tag. Tags inside its strings are just characters.
    start = len(after) - len(after.lstrip())
    try:
        obj, end = _DECODER.raw_decode(after, start)
    except ValueError as exc:
        return ParsedResponse(text=before.strip(), error=f"The tool call may be cut off and is not valid JSON ({exc}). {HOW_TO_FIX}")

    # The closing tag must come right after the JSON, and no second call may follow.
    tail = after[end:].lstrip()
    if not tail.startswith(END):
        return ParsedResponse(text=before.strip(), error=f"The tool call has no closing {END} tag right after its JSON, so it may be cut off. {HOW_TO_FIX}")
    rest = tail[len(END):]
    if START in rest:
        return ParsedResponse(text=before.strip(), error=f"The reply contains more than one tool call. {HOW_TO_FIX}")

    name = obj.get("name") if isinstance(obj, dict) else None
    args = obj.get("args", obj.get("arguments", {})) if isinstance(obj, dict) else None
    if not isinstance(name, str) or not isinstance(args, dict):
        return ParsedResponse(text=before.strip(), error=f'A tool call needs a string "name" and an object "args". {HOW_TO_FIX}')

    # The reply's text is everything outside the call.
    text = "\n\n".join(part for part in (before.strip(), rest.strip()) if part)
    return ParsedResponse(text=text, tool_call=ToolCall(name, args))

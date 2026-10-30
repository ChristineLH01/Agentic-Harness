"""Problem 7 (agent_loop): the loop that alternates model calls and tool calls.

The model's context is a list of system/user/assistant messages. A tool result
or an error goes back to the model as a user message. A reply with no tool call
is the final answer, except an empty one or one that only announces an action
("I'll look that up."): the loop sends it back as an error counted as malformed.
Before each model call, compaction.py shortens the context if the call would not
fit config.context_budget (Problem 11).
"""

from __future__ import annotations

import copy
import re

from cs329z_hw1.agent.compaction import RESULT_SHARE, compact, payload, shorten
from cs329z_hw1.agent.tool_registry import ToolRegistry
from cs329z_hw1.agent.tool_protocol import format_tool_result, parse_response, render_tools
from cs329z_hw1.llm import OutputTruncated
from cs329z_hw1.tokens import count_message_tokens, count_tokens
from cs329z_hw1.types import AgentConfig, AgentResult, ToolResult

MAX_MALFORMED = 3

SYSTEM_PROMPT = """\
You are Cardinal, the assistant for employees of Cardinal Energy. Questions are \
about Cardinal Energy: its people, policies, products and emails. The tools under \
"## Tools" look these up; "## How to call a tool" explains how to call them.

- Do not answer from memory or make things up. Look facts up with the tools, and try \
other words before saying you could not find something.
- To use a tool, write the call block in your reply text; there is no built-in function \
calling. A reply without a call is taken as your final answer, so never announce or \
offer an action without doing it, such as "I'll look that up.": make the call instead.
- Tool results appear in the conversation as user messages of this form:
  <tool_result tool="TOOL_NAME" status="STATUS">
  RESULT
  </tool_result>
  STATUS "ok" means success. Any other status means the call failed and RESULT \
explains why: fix the call and try again.
- When you have the answer, reply in plain text without any tool call tags."""

# A reply with no tool call that is empty or only announces an action ("I'll look
# that up.") is not an answer. It is sent back with INVALID_ANSWER.
ANNOUNCEMENT = re.compile(
    r"(I['’]ll|I will|I['’]m going to|I am going to|I['’]m (checking|looking|searching)|"
    r"Let me|Let['’]s|Searching|Checking|Looking)\b", re.I)
INVALID_ANSWER = ("Your reply has no tool call and no answer. There is no built-in function calling: "
                  "write the call block in your reply text, as described under \"## How to call a tool\", "
                  "or give your final answer.")


class AgentSession:
    def __init__(self, lm, tools, config: AgentConfig):
        self.lm = lm
        self.aux_lm = config.aux_lm or lm
        self.config = config
        self.mode = config.mode
        self.registry = ToolRegistry(tools)
        self.transcript: list[dict] = []
        self.system = f"{SYSTEM_PROMPT}\n\n{render_tools(self.registry.schemas())}"
        # The messages after the system prompt, each with its transcript event (see compaction.py),
        # and the summary of the turns that compaction replaced.
        self.context: list[dict] = []
        self.summary = ""

    # Used from Problem 13 (interactive) on.
    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def send(self, user_message: str) -> AgentResult:
        self._add("user", user_message, type="user", content=user_message)
        turns = tokens = malformed = 0

        while True:
            if turns >= self.config.max_turns:
                return self._stop("max_turns", f"Stopped after {turns} model calls without a final answer.")
            if tokens >= self.config.token_budget:
                return self._stop("token_budget", f"Stopped after using {tokens} tokens without a final answer.")

            turns += 1
            messages = self._messages()
            sent = count_message_tokens(messages)
            try:
                reply = self.lm(messages)
            except OutputTruncated as exc:
                tokens += sent + count_tokens(exc.text)
                self._error("Your reply was cut off because it was too long. Write a shorter reply.")
                malformed += 1
            else:
                tokens += sent + count_tokens(reply)
                if reply.strip():
                    self._add("assistant", reply, type="assistant", content=reply)
                else:
                    # Logged only: kept empty reply out of the context so the model does not copy it.
                    self.transcript.append({"type": "assistant", "content": reply})
                parsed = parse_response(reply)
                if parsed.error:
                    # The call could not be parsed.
                    self._error(parsed.error)
                    malformed += 1
                elif parsed.tool_call is None and (not parsed.text or ANNOUNCEMENT.match(parsed.text)):
                    # No call, and the reply is empty or only announces an action.
                    self._error(INVALID_ANSWER)
                    malformed += 1
                elif parsed.tool_call is None:
                    # No call: the reply is the final answer. Compaction keeps it for later turns.
                    self.context[-1]["answer"] = True
                    return self._stop("done", parsed.text)
                else:
                    # A call: run it. An unknown tool or invalid arguments is malformed.
                    status = self._run(parsed.tool_call)
                    malformed = malformed + 1 if status in ("unknown_tool", "invalid_args") else 0

            if malformed >= MAX_MALFORMED:
                return self._stop("malformed", f"Stopped after {malformed} malformed replies in a row.")

    def _add(self, role: str, message: str, **event) -> None:
        """Append one message to the model's context and one event to the transcript."""
        self.context.append({"role": role, "content": message, "event": event})
        self.transcript.append(event)

    def _messages(self) -> list[dict]:
        """The messages of the next model call, compacted first if they would not fit."""
        self.summary = compact(self.system, self.summary, self.context, self.config.context_budget, self.aux_lm)
        return payload(self.system, self.summary, self.context)

    def _stop(self, status: str, text: str) -> AgentResult:
        """Return the final result, with a copy of the transcript so far."""
        return AgentResult(text=text, status=status, transcript=copy.deepcopy(self.transcript))

    def _error(self, error: str) -> None:
        """Show an error to the model and log it."""
        self._add("user", error, type="error", content=error)
        
    def _run(self, call) -> str:
        """Run one tool call, show its result to the model, and return its status."""
        self.transcript.append({"type": "tool_call", "name": call.name, "args": call.args})
        result = self.registry.execute(call.name, call.args)
        # A result too long for the context is cut before the model or the transcript sees it
        result = ToolResult(result.status, shorten(result.content, self.config.context_budget // RESULT_SHARE))
        self._add("user", format_tool_result(call.name, result),
                  type="tool_result", name=call.name, status=result.status, content=result.content)
        return result.status

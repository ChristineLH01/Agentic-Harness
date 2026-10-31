"""Problem 13 (interactive): which calls wait for the user's approval.

Policy: in confirm mode, a valid call to a tool that is not read-only (of the
Cardinal tools, only run_terminal) runs only after config.user approves it.
Read-only tools never ask, ask_user, remember and recall included, so the user
is asked only about calls that can change something. In auto mode nothing asks.
Steering is in agent_loop.py and the ask_user tool in cardinal_tools.py.
"""

from __future__ import annotations

from cs329z_hw1.types import Approval, ToolResult


def denial(registry, call, mode: str, user) -> ToolResult | None:
    """The "denied" result of a call the user does not approve, or None if it may run.
    The loop validates the call first, so a call the registry rejects is never sent for approval."""
    if mode != "confirm" or registry.tools[call.name].read_only:
        return None
    approval = user.approve(call) if user is not None else Approval(False, "No user is available to approve it.")
    if approval.approved:
        return None
    reason = approval.reason.strip() or "The user gave no reason."
    return ToolResult("denied", f"The user did not approve this call, so it did not run. Reason: {reason} "
                                "Do not repeat the call. Change your plan to fit the reason, or answer without it.")

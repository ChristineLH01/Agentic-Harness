"""Problem 1 (priority): Mark email priority"""

from __future__ import annotations

import re
from datetime import datetime

from cs329z_hw1.llm import BudgetExceeded, OutputTruncated
from cs329z_hw1.types import Email, LMCallable

SYSTEM_PROMPT = """\
Given one email and its metadata, classify its priority for the recipient as \
urgent, normal, or ignore, with a one-sentence reason.

Judge the email from the point of view of the person it is addressed to (the \
To field), at the time it was sent. The question is what that person should \
do with the message, not how important the topic is to the company.

## urgent

The recipient has to act or decide soon. Two things must both be true:

1. The email asks the recipient to do something, or reports a live problem \
that the recipient is responsible for (an outage, a failed payment, a customer \
or manager escalating an issue).
2. There is time pressure: the email states or clearly implies that the action \
is needed the same day, the next working day, or by a stated deadline no more \
than two working days away. A working day is Monday to Friday, counted from \
the email's date.

Examples:
- "The bank rejected this morning's wire. I need you to approve the corrected \
one by 1:00 PM or we pay a late penalty."
- "The customer's CFO is threatening to cancel. Can you call him before 4:00 today?"

## normal

Legitimate work or personal mail that was written for this recipient or for a \
small working group, with no deadline in the next two working days. This \
covers status updates and FYIs, ordinary questions, documents sent for review \
without a near deadline, meeting scheduling for later dates, replies in an \
ongoing discussion, and personal notes from friends or family.

Examples:
- "Attached is the draft term sheet. Let me know if you have comments before \
our meeting the week after next."
- "Thanks for lunch yesterday. Are you still coming to the lake in August?"

## ignore

Mail the recipient can leave unread at no cost: advertising and spam from \
vendors, newsletters and news digests, announcements sent to large employee \
lists (events, training offerings, facilities notices), and automated \
notifications that do not require the recipient to do anything.

Examples:
- "This week only: 40% off all standing desks. Reply REMOVE to unsubscribe."
- "To all employees: the annual bake sale is in the lobby on Friday from 11:00 to 1:00."

## Deciding between two labels

- The wording does not decide the label. An email with "URGENT" in the subject \
that asks nothing of the recipient is not urgent. A calm email that says "I \
need your signature by 3:00 PM today" is urgent.
- A request with a deadline more than two working days away, or with no \
deadline, is normal.
- Mass mailings and advertising are ignore even when they use words such as \
"act now" or "important", because nothing is asked of this recipient in particular.
- An automated notice that asks this specific recipient to act (for example, \
an expense report waiting for their approval) is normal, or urgent if it \
states a deadline within two working days.
- A short personal or social message from someone the recipient knows is \
normal, not ignore.

## Reply format

Reply with exactly these two lines and nothing else:
category: <urgent, normal or ignore>
reason: <one sentence>"""

def classify_priority(email: Email, lm: LMCallable) -> dict:
    """Return {"category": urgent|normal|ignore|unknown, "reason": str}."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_email(email)},
    ]
    try:
        reply = lm(messages)
    except OutputTruncated as exc:
        reply = exc.text or ""
    except BudgetExceeded:
        raise
    except Exception as exc:  # label unknown for a failed call
        return {"category": "unknown", "reason": f"The model call failed: {exc}"}
    return parse_reply(reply)


def render_email(email: Email) -> str:
    """The email as the model sees it: headers with the weekday, then a bounded body."""
    to = list(email.get("to") or [])
    # Truncate the long To list with a total count.
    shown = ", ".join(to[:5])
    if len(to) > 5:
        shown += f", ... ({len(to)} recipients in total)"

    body = email.get("body") or ""
    # Truncate long body, assuming the initial part can decide priority.
    if len(body) > 5000:
        body = body[:5000] + "\n[... rest of the email cut]"
    return (
        f"Date: {_date_with_weekday(email.get('date') or '')}\n"
        f"From: {email.get('from', '')}\n"
        f"To: {shown or '(no recipients listed)'}\n"
        f"Subject: {email.get('subject') or '(no subject)'}\n\n"
        f"{body.strip() or '(empty body)'}"
    )

def _date_with_weekday(date: str) -> str:
    """Reformat the date with the weekday, or "(unknown)" if it can't be parsed."""
    try:
        parsed = datetime.strptime(date[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return date or "(unknown)"
    return f"{parsed:%Y-%m-%d} ({parsed:%A}) {parsed:%H:%M} UTC"

_CATEGORY_LINE = re.compile(r"^\W*category\W*:\W*(urgent|normal|ignore)\b", re.IGNORECASE | re.MULTILINE)
_REASON_LINE = re.compile(r"^\W*reason\W*:[\s*_]*(.+)$", re.IGNORECASE | re.MULTILINE)

def parse_reply(reply: str) -> dict:
    """Parse the format "category:" and "reason:" lines."""
    text = (reply or "").strip()
    match = _CATEGORY_LINE.search(text)
    if not match:
        shown = text[:100] + ("..." if len(text) > 100 else "")
        return {"category": "unknown", "reason": f"Failed to parse a category from the reply: {shown!r}"}
    reason_match = _REASON_LINE.search(text)
    reason = reason_match.group(1).strip() if reason_match else " ".join(text.split())
    return {"category": match.group(1).lower(), "reason": reason or "The model gave no reason."}

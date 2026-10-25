"""Problem 2 (daily_digest): classify priority and write summary of daily emails."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from cs329z_hw1.llm import BudgetExceeded, OutputTruncated
from cs329z_hw1.pipelines.email_priority import classify_priority
from cs329z_hw1.types import Email, LMCallable

CLASSIFY_WORKERS = 8  # concurrent calls for priority classification
URGENT_EXCERPT_CHARS = 800  # body length shown for an urgent item
NORMAL_EXCERPT_CHARS = 200  # body length shown for a normal item
WORD_LIMIT = 275  # limit = 300, ask for 275 to avoid model overshoot.

SYSTEM_PROMPT = """\
Given one day's emails, write the morning digest for a busy employee, the \
recipient. Each email carries a priority label, URGENT or NORMAL, and the \
urgent ones are listed first.

Write the digest as follows:
1. Must keep it under {word_limit} words in total.
2. Cover the URGENT emails first. You must name every urgent email: give the company, \
project, site or person named in each urgent email, using the proper names from its subject \
or body.
3. Then summarize the NORMAL emails at a high level, grouping related ones and \
leaving out minor details.
4. Reply with the digest only, without headings or commentary."""


def write_daily_digest(emails: list[Email], lm: LMCallable) -> str:
    """Return summary digest for one day's emails as a string."""
    if not emails:
        return "No emails were received on this day."

    with ThreadPoolExecutor(max_workers=CLASSIFY_WORKERS) as pool:
        labels = list(pool.map(lambda e: classify_priority(e, lm)["category"], emails))

    urgent, normal = _group_by_thread(emails, labels)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(word_limit=WORD_LIMIT)},
        {"role": "user", "content": _render_day(emails, urgent, normal)},
    ]
    try:
        digest = lm(messages).strip()
    except OutputTruncated as exc:
        digest = (exc.text or "").strip()
    except BudgetExceeded:
        raise
    except Exception as exc:
        return f"Digest not written: the model call failed: {exc}"
    return digest or "Digest not written: the model returned an empty reply."


def _group_by_thread(emails: list[Email], labels: list[str]) -> tuple[list[dict], list[dict]]:
    """One item per thread, represented by the earliest email among the ones with
    the highest-priority label in the thread (assuming the first message states the matter
    in full; follow-ups are often short). ``unknown`` counts as normal; ``ignore``
    threads are dropped."""
    rank = {"urgent": 2, "normal": 1, "unknown": 1, "ignore": 0}
    threads: dict[str, dict] = {}
    for email, label in zip(emails, labels):
        item = threads.get(email["thread_id"])
        if item is None:
            threads[email["thread_id"]] = {"rank": rank[label], "email": email}
        elif rank[label] > item["rank"] or (rank[label] == item["rank"] and _earlier(email, item["email"])):
            item["rank"], item["email"] = rank[label], email
    items = list(threads.values())
    urgent = [i for i in items if i["rank"] == 2]
    normal = [i for i in items if i["rank"] == 1]
    return urgent, normal


def _earlier(a: Email, b: Email) -> bool:
    """True when ``a`` has a date and it is earlier than ``b``'s. An email without
    a date never replaces one that has a date; with neither dated, the first seen stays."""
    date_a, date_b = a.get("date") or "", b.get("date") or ""
    return bool(date_a) and (not date_b or date_a < date_b)


def _render_day(emails: list[Email], urgent: list[dict], normal: list[dict]) -> str:
    """Prepare model input: return a string that lists the day's emails."""
    date = (emails[0].get("date") or "")[:10]
    try:
        day = f"{date} ({datetime.strptime(date, '%Y-%m-%d'):%A})"
    except ValueError:
        day = date or "(unknown date)"
    lines = [f"Emails of {day} ({len(urgent)} urgent, {len(normal)} normal):", ""]
    lines += [_render_item(i, "URGENT", URGENT_EXCERPT_CHARS) for i in urgent]
    lines += [_render_item(i, "NORMAL", NORMAL_EXCERPT_CHARS) for i in normal]
    return "\n".join(lines)


def _render_item(item: dict, label: str, excerpt_chars: int) -> str:
    """Return a short string describing one thread's representative email."""
    email = item["email"]
    body = " ".join((email.get("body") or "").split())
    # Truncate long body, assuming the initial part carries the key information.
    if len(body) > excerpt_chars:
        body = body[:excerpt_chars] + " [...]"
    return (
        f"- From: {email.get('from', '')}\n"
        f"  Priority: {label}\n"
        f"  Subject: {email.get('subject') or '(no subject)'}\n"
        f"  Body: {body or '(empty body)'}"
    )

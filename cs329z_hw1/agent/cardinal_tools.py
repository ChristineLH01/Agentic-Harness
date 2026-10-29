"""Problems 8: the Cardinal tools, each with its description and implementation.

- Email tools, built on the Part 1 code: search_emails and read_email,
  email_qa and daily_digest.
- Document tools: search_docs and read_doc, over the index in search_docs.py (P9).
- Tools of later problems: run_terminal (P10), remember and recall (P12), and
  ask_user (P13).
"""

from __future__ import annotations

import functools

from cs329z_hw1 import data
from cs329z_hw1.agent.search_docs import doc_index
from cs329z_hw1.cardinal import CARDINAL_TOOLS, cardinal_spec
from cs329z_hw1.pipelines.daily_digest import write_daily_digest
from cs329z_hw1.pipelines.bm25 import BM25Index
from cs329z_hw1.pipelines.email_qa import answer_question

######## Tool Descriptions ########
DESCRIPTIONS = {
    "search_emails": (
        "Search the company email archive by keywords. Returns up to k emails, best match "
        "first, each with its id, date, sender, subject and the start of its body. "
        "Use words likely to appear in the email for search. When a result looks relevant but "
        "its snippet does not hold the answer, read the whole email with read_email."
    ),
    "read_email": "Read one email by its id, for example \"em-00042\". Very long bodies are cut.",
    "email_qa": (
        "Answer a factual question based on the email archive. It runs its own searches "
        "and returns an answer with the ids of the supporting emails, or says that it "
        "found none."
    ),
    "daily_digest": (
        "Write the summary digest of all emails sent on one UTC date (YYYY-MM-DD): "
        "naming every urgent email first, then a short summary of the normal or unknown emails, skipping ignored emails."
    ),
    "search_docs": (
        "Search the Cardinal Energy company documents by keywords. Returns top k related passages, "
        "each with its doc_id, line number, title, date, section and a short snippet. "
        "When a result looks relevant but its snippet does not hold the answer, read the "
        "document around that line with read_doc. "
        "Search matches exact words, so if a search misses, try synonyms or related words."
    ),
    "read_doc": (
        "Read a window of one company document from a start_line that a search_docs result gives, or a few lines "
        "earlier for context. Returns the document's title and date, the line range shown "
        "out of the total, and the text. "
        "Announcements and memos often change older policies: when documents disagree, "
        "follow the one with the newest date."
    ),
    "run_terminal": "Run a shell command in a sandboxed workspace and return its output.",
    "remember": "Save a fact to long-term memory so that later conversations can use it.",
    "recall": "Search long-term memory for facts saved earlier.",
    "ask_user": "Ask the user a question and get their answer. Use it when the request is unclear.",
}

######## The email tools wrap the Part 1 pipelines ########
@functools.lru_cache(maxsize=1)
def _email_index():
    """The emails, their BM25 index and an id-to-email map. Built the first time a
    tool needs it and shared by every session."""
    emails = data.load_emails()
    return emails, BM25Index([data.email_text(e) for e in emails]), {e["id"]: e for e in emails}

MAX_EMAIL_K = 10  # max number of searched email returned at once

def _search(query: str, k: int = 5) -> list[dict]:
    """The best k emails for query, at most MAX_EMAIL_K."""
    emails, index, _ = _email_index()
    return [emails[i] for i, _ in index.search(query, min(k, MAX_EMAIL_K))]


def search_emails(query: str, k: int = 5) -> str:
    """One summary per matching email, best match first."""
    results = _search(query, k)
    if not results:
        return f"No emails match {query!r}. Try other keywords."
    return "\n".join(_summary(e) for e in results)

SNIPPET_CHARS = 800  # start of the email body shown for model per search result

def _summary(email: dict) -> str:
    """The email's id, date, sender, subject and the start of its body."""
    body = " ".join(email["body"].split())
    snippet = body[:SNIPPET_CHARS] + ("..." if len(body) > SNIPPET_CHARS else "")
    return (f"- ID: {email['id']}\n  Date: {email['date']}\n  From: {email['from']}\n"
            f"  Subject: {email['subject']}\n  Snippet: {snippet}")

READ_CHARS = 6000  # longest body read_email shows

def read_email(email_id: str) -> str:
    """The whole email with its headers. A body longer than READ_CHARS is cut."""
    email = _email_index()[2].get(email_id)
    if email is None:
        raise LookupError(f"No email has the id {email_id!r}.")
    body = email["body"].strip()
    if len(body) > READ_CHARS:
        body = body[:READ_CHARS] + f"\n[... {len(body) - READ_CHARS} more characters not shown]"
    return (f"ID: {email['id']}\nThread: {email['thread_id']}\nDate: {email['date']}\n"
            f"From: {email['from']}\nTo: {', '.join(email['to'])}\nSubject: {email['subject']}\n\n{body}")


######## Document tools: search and read over the document index (search_docs.py) ########
MAX_DOC_K = 20  # max results search_docs returns at once

def search_docs(query: str, k: int = 5) -> str:
    results = doc_index().search(query, min(k, MAX_DOC_K))
    if not results:
        return f"No documents match {query!r}. Try other words."
    # Snippet lines are indented, so a snippet line starting with "- " does not look like a new result.
    return "\n".join(f"- {r['doc_id']}, line {r['line']} ({r['title']}; {r['date']})\n"
                     f"  Section: {r['section']}\n  " + r["snippet"].replace("\n", "\n  ") for r in results)


def read_doc(doc_id: str, start_line: int = 1) -> str:
    index = doc_index()
    w = index.read(doc_id, start_line)
    end, total = w["end_line"], w["total_lines"]
    # The footer tells the model where the next window starts.
    more = f"[Lines {end + 1}-{total} not shown. Read on with start_line={end + 1}.]" if end < total else "[End of document.]"
    return (f"{doc_id} ({index.docs[doc_id]['title']}; {index.dates[doc_id]}), lines {start_line}-{end} of {total}:\n\n"
            f"{w['text']}\n\n{more}")


def _not_ready(name: str):
    """A stand-in for a tool of a later problem. Calling it gives a tool_error."""
    def fn(**args):
        raise NotImplementedError(f"{name} is not available yet.")
    return fn

######## Assemble the toolset ######## 
def cardinal_tools(config, aux_lm) -> list:
    """All ten Cardinal tools. The tools defined here need config or aux_lm, and
    their LLM calls go to aux_lm."""

    def email_qa(question: str) -> str:
        result = answer_question(question, _search, aux_lm)
        if result["answer"] is None:
            return "No answer was found in the emails."
        return f"Answer: {result['answer']}\nSupport: {', '.join(result['support']) or 'none'}"

    def daily_digest(date: str) -> str:
        return write_daily_digest(data.emails_on(date), aux_lm)

    # Asks config.user, and fails when the session has no user.
    def ask_user(question: str) -> str:
        if config.user is None:
            raise RuntimeError("No user is available to answer questions.")
        return config.user.ask(question)

    # Tools without an implementation yet get the _not_ready stand-in.
    fns = {"search_emails": search_emails, "read_email": read_email, "email_qa": email_qa,
           "daily_digest": daily_digest, "search_docs": search_docs, "read_doc": read_doc, "ask_user": ask_user}
    return [cardinal_spec(name, DESCRIPTIONS[name], fns.get(name) or _not_ready(name)) for name in CARDINAL_TOOLS]


def session_tools(tools, config, aux_lm) -> list:
    """The tools a session gets. tools=None gives all ten Cardinal tools. An explicit
    list gets exactly those tools, plus ask_user if config.user is set and remember
    and recall if config.memory_dir is set."""
    if tools is None:
        return cardinal_tools(config, aux_lm)
    extra = ({"ask_user"} if config.user is not None else set()) | (
        {"remember", "recall"} if config.memory_dir is not None else set())
    return list(tools) + [t for t in cardinal_tools(config, aux_lm) if t.name in extra]

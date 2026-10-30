"""Problem 12 (memory): facts that persist across sessions.

The store is one JSON file, memory_dir/memory.json: a list of facts, each one
sentence, oldest first. remember adds a fact, and recall searches the facts with
BM25. A new session starts with the newest facts in its system prompt, up to a
size limit, so the model knows what it has; recall also finds the older ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from cs329z_hw1.pipelines.bm25 import BM25Index
from cs329z_hw1.tokens import count_tokens

MEMORY_SHARE = 10  # the facts in the system prompt use at most 1/MEMORY_SHARE of the budget
RECALL_K = 10  # most facts one recall returns


class MemoryStore:
    def __init__(self, memory_dir) -> None:
        self.path = Path(memory_dir) / "memory.json"

    def facts(self) -> list[str]:
        """Every saved fact, oldest first."""
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def remember(self, fact: str) -> str:
        """Add one fact to the file, unless it is already there."""
        fact = " ".join(fact.split())
        if not fact:
            raise ValueError("The fact is empty.")
        facts = self.facts()
        if fact.lower() in (f.lower() for f in facts):
            return f"Already saved: {fact}"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(facts + [fact], indent=2, ensure_ascii=False), encoding="utf-8")
        return f"Saved: {fact}"

    def recall(self, query: str) -> str:
        """The facts that best match query, in the order they were saved."""
        facts = self.facts()
        if not facts:
            return "Memory is empty: no session has saved a fact yet."
        hits = sorted(i for i, _ in BM25Index(facts).search(query, RECALL_K))
        if not hits:
            return f"No saved fact matches {query!r}. Try other words."
        return "Saved facts, oldest first (a newer fact overrides an older one):\n" + "\n".join(f"- {facts[i]}" for i in hits)

    def prompt(self, max_tokens: int) -> str:
        """The memory section of the system prompt: the newest facts that fit in max_tokens."""
        facts, shown, used = self.facts(), [], 0
        for fact in reversed(facts):
            used += count_tokens(fact) + 1
            if used > max_tokens:
                break
            shown.insert(0, f"- {fact}")
        hidden = len(facts) - len(shown)
        lines = shown or ["(No facts are saved yet.)"]
        if hidden:
            lines.append(f"[{hidden} older facts are not shown here; find them with recall.]")
        return ("\n\n## Memory\nYour memory persists across sessions: a new session knows only the facts that "
                "other sessions saved with remember. So when the user tells you something about themselves "
                "or how they want things done, call remember first, then answer.\nFacts saved in other "
                "sessions, oldest first (a newer fact overrides an older one). Use them to understand the "
                "user's requests, for example what they mean by a term:\n" + "\n".join(lines))

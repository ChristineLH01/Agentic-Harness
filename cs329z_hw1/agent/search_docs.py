"""Problem 9 (search_docs): the document index behind the search_docs and read_doc
tools in cardinal_tools.py.

- Build a BM25 index over sections: a heading and the lines under it, up to the next heading.
- Search result points to the line where its snippet starts. The snippet is placed where it
holds the most of the query's rare words.
- Read returns a window of whole lines from start_line.
"""

from __future__ import annotations

import functools
import re

from cs329z_hw1 import data
from cs329z_hw1.pipelines.bm25 import BM25Index
from cs329z_hw1.tokenizer import tokenize

SNIPPET_CHARS = 600  # 150 tokens shown for each search result
WINDOW_CHARS = 4000  # 1000 tokens per read_doc window


class DocIndex:
    def __init__(self, docs: list[dict]) -> None:
        self.docs = {d["doc_id"]: d for d in docs}
        self.lines = {d["doc_id"]: d["text"].splitlines() for d in docs}
        self.dates = {doc_id: _date(lines[:10]) for doc_id, lines in self.lines.items()}
        # A section is (doc_id, first line, end line), starting at line 0 and at every heading.
        self.sections = []
        for doc_id, lines in self.lines.items():
            # Lines a section starts
            bounds = sorted({0, *(i for i, line in enumerate(lines) if line.startswith("#"))}) + [len(lines)]
            self.sections += [(doc_id, sec_start, sec_end) for sec_start, sec_end in zip(bounds, bounds[1:])
                              if sec_start < sec_end]
        # The title is added to each section, so its words match every section of the document.
        self.bm25 = BM25Index([self.docs[doc_id]["title"] + "\n" + "\n".join(self.lines[doc_id][sec_start:sec_end])
                               for doc_id, sec_start, sec_end in self.sections])

    def search(self, query: str, k: int = 5) -> list[dict]:
        terms = set(tokenize(query))
        results = []
        # BM25 picks the sections; the snippet is then placed inside each one.
        for i, _ in self.bm25.search(query, k):
            doc_id, sec_start, sec_end = self.sections[i]
            lines = self.lines[doc_id]
            # The snippet with the most weight of query terms within the section, earliest first.
            best = max((j for j in range(sec_start, sec_end) if lines[j].strip()),
                       key=lambda j: (self._weight(_snippet(lines, j, sec_end), terms), -j))
            results.append({
                "doc_id": doc_id,
                "title": self.docs[doc_id]["title"],
                "snippet": _snippet(lines, best, sec_end),
                "line": best + 1,  # 1-based like read_doc
                "section": (lines[sec_start].lstrip("# ") if lines[sec_start].startswith("#")
                            else self.docs[doc_id]["title"]),
                "date": self.dates[doc_id],
            })
        return results

    def _weight(self, text: str, terms: set[str]) -> float:
        """The idf of the query terms in text, so rare words count more."""
        return sum(self.bm25.idf.get(t, 0.0) for t in terms & set(tokenize(text)))

    def read(self, doc_id: str, start_line: int = 1) -> dict:
        if doc_id not in self.lines:
            raise LookupError(f"No document has the doc_id {doc_id!r}. Use search_docs to find doc_ids.")
        lines = self.lines[doc_id]
        if not 1 <= start_line <= len(lines):
            raise ValueError(f"start_line must be between 1 and {len(lines)} for {doc_id}.")
        # Add whole lines until the next one would pass WINDOW_CHARS. A line is never cut.
        end, chars = start_line, len(lines[start_line - 1])  # end is the 1-based last line
        while end < len(lines) and chars + 1 + len(lines[end]) <= WINDOW_CHARS:
            chars += 1 + len(lines[end])
            end += 1
        return {"doc_id": doc_id, "start_line": start_line, "end_line": end,
                "total_lines": len(lines), "text": "\n".join(lines[start_line - 1:end])}


def _snippet(lines: list[str], start: int, end: int) -> str:
    """Lines start, start + 1, ... before end, up to SNIPPET_CHARS characters. Blank lines are skipped."""
    snippet, j = lines[start], start + 1
    while len(snippet) < SNIPPET_CHARS and j < end:
        if lines[j].strip():
            snippet += "\n" + lines[j]
        j += 1
    return snippet[:SNIPPET_CHARS]

DATE = re.compile(r"^(Effective|Posted|Date|Updated|Last updated|As of): (\S+)", re.M)

def _date(header: list[str]) -> str:
    """The date in the header"""
    found = dict(DATE.findall("\n".join(header)))
    label = "Effective" if "Effective" in found else next(iter(found), None)
    return f"{label} {found[label]}" if label else "undated"


# Built the first time a tool needs it, and shared by every session.
@functools.lru_cache(maxsize=1)
def doc_index() -> DocIndex:
    return DocIndex(data.load_docs())

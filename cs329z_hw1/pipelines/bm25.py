"""Problem 3 (bm25): BM25 retrieval over an inverted index."""

from __future__ import annotations

import heapq
import math
from collections import Counter

from cs329z_hw1.tokenizer import tokenize

K1 = 1.5
B = 0.75


class BM25Index:
    """Inverted index: for each term, the (doc_id, term frequency) pairs of the
    documents containing it, plus the per-document numbers BM25 needs."""

    def __init__(self, docs: list[str]) -> None:
        self.postings: dict[str, list[tuple[int, int]]] = {}
        lengths = []
        for doc_id, doc in enumerate(docs):
            tokens = tokenize(doc)
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, []).append((doc_id, tf))
        n = len(docs)
        avgdl = sum(lengths) / max(n, 1) or 1.0  # 1.0 only when every document is empty
        # Length normalization per document, so a query does no division.
        self.norm = [K1 * (1 - B + B * length / avgdl) for length in lengths]
        self.idf = {
            term: math.log((n - len(hits) + 0.5) / (len(hits) + 0.5) + 1)
            for term, hits in self.postings.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top-k (doc_id, score), best first, ties broken by lower doc_id.
        Only the postings of the query's distinct terms are read."""
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for doc_id, tf in self.postings[term]:
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (K1 + 1) / (tf + self.norm[doc_id])
        return heapq.nsmallest(k, scores.items(), key=lambda pair: (-pair[1], pair[0]))

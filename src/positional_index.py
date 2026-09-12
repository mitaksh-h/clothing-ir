"""
positional_index.py
--------------------
Part C: Extend the inverted index to a positional index.

Conceptual structure (as specified in the assignment):
    term -> df -> [(docID, tf, [p1, p2, ...]), ...]

Positions are 0-based indices into the pre-processed (stemmed,
stop-word-removed) token stream of each document -- the SAME token
stream used to build the Part A/B inverted index, so that phrase and
proximity results are directly comparable with plain VSM results.

This module supports:
    - Exact phrase search ("cotton shirt")
    - Ordered proximity search (term1 WITHIN/k term2)
"""

from collections import defaultdict
from typing import Dict, List, Tuple
from preprocess import preprocess


class PositionalIndex:
    def __init__(self):
        # term -> {docid: [positions]}
        self.index: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
        self.df: Dict[str, int] = defaultdict(int)
        self.doc_titles: Dict[str, str] = {}
        self.doc_categories: Dict[str, str] = {}
        self.N: int = 0

    def build(self, documents: List[Dict[str, str]]):
        self.N = len(documents)
        for doc in documents:
            docid = doc["docid"]
            self.doc_titles[docid] = doc["title"]
            self.doc_categories[docid] = doc["category"]

            tokens = preprocess(doc["content"])
            positions: Dict[str, List[int]] = defaultdict(list)
            for pos, tok in enumerate(tokens):
                positions[tok].append(pos)

            for term, pos_list in positions.items():
                self.index[term][docid] = pos_list

        for term, postings in self.index.items():
            self.df[term] = len(postings)

    def get_positions(self, term: str) -> Dict[str, List[int]]:
        return self.index.get(term, {})

    # ------------------------------------------------------------------
    # Phrase search: "term1 term2 ... termk" must occur CONSECUTIVELY.
    # ------------------------------------------------------------------
    def phrase_search(self, phrase: str) -> List[Tuple[str, List[int]]]:
        """
        Exact phrase search using positions.
        Returns a list of (docid, [start_positions]) for documents where
        the phrase occurs as consecutive tokens (after the same
        stemming/stop-word pipeline as indexing).
        """
        query_terms = preprocess(phrase)
        if not query_terms:
            return []

        # Documents that contain the first term are the candidate set.
        first_term_postings = self.get_positions(query_terms[0])
        if not first_term_postings:
            return []

        results = []
        for docid, first_positions in first_term_postings.items():
            matches = []
            for start_pos in first_positions:
                ok = True
                for i, term in enumerate(query_terms[1:], start=1):
                    term_positions = self.get_positions(term).get(docid, [])
                    if (start_pos + i) not in term_positions:
                        ok = False
                        break
                if ok:
                    matches.append(start_pos)
            if matches:
                results.append((docid, matches))

        # Sort by docid for determinism
        results.sort(key=lambda x: x[0])
        return results

    # ------------------------------------------------------------------
    # Ordered proximity search: term1 must occur within k positions
    # BEFORE term2 (term1 ... term2, gap <= k). This matches the
    # assignment's "term1 WITHIN/k term2" notation.
    # ------------------------------------------------------------------
    def proximity_search(self, term1: str, term2: str, k: int) -> List[Tuple[str, List[Tuple[int, int]]]]:
        """
        Ordered proximity search: term1 occurs at position p1 and term2
        occurs at position p2, with p1 < p2 and (p2 - p1) <= k + 1
        (i.e. at most k tokens in between).

        Returns a list of (docid, [(pos1, pos2), ...]) for matching pairs.
        """
        t1 = preprocess(term1)
        t2 = preprocess(term2)
        if not t1 or not t2:
            return []
        t1, t2 = t1[0], t2[0]

        postings1 = self.get_positions(t1)
        postings2 = self.get_positions(t2)

        results = []
        common_docs = set(postings1.keys()) & set(postings2.keys())
        for docid in common_docs:
            pairs = []
            for p1 in postings1[docid]:
                for p2 in postings2[docid]:
                    if 0 < (p2 - p1) <= (k + 1):
                        pairs.append((p1, p2))
            if pairs:
                results.append((docid, sorted(pairs)))

        results.sort(key=lambda x: x[0])
        return results

    def save_positional_report(self, path: str, top_n_postings: int = 8):
        with open(path, "w", encoding="utf-8") as f:
            f.write("POSITIONAL INDEX (Part C)\n")
            f.write(f"Total documents (N): {self.N}\n")
            f.write(f"Vocabulary size: {len(self.index)}\n")
            f.write("=" * 70 + "\n\n")
            for term in sorted(self.index.keys()):
                postings = sorted(self.index[term].items())
                shown = postings[:top_n_postings]
                more = f" ... (+{len(postings) - top_n_postings} more)" if len(postings) > top_n_postings else ""
                f.write(f"{term:20s} df={self.df[term]:3d}\n")
                for docid, positions in shown:
                    f.write(f"    {docid}: tf={len(positions)}  positions={positions}\n")
                if more:
                    f.write(f"   {more}\n")
                f.write("\n")

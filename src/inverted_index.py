"""
inverted_index.py
------------------
Part A: Build an inverted index (dictionary) with term frequencies (tf)
and document frequencies (df) from the pre-processed corpus.

Conceptual structure:
    term -> df -> [(docID, tf), (docID, tf), ...]

We store this as:
    self.index: Dict[str, Dict[str, int]]   # term -> {docid: tf}
    self.df:    Dict[str, int]              # term -> document frequency

We also keep the raw (un-summed) token stream per document, which the
positional index (Part C) builds on top of.
"""

from collections import defaultdict
from typing import Dict, List
from preprocess import preprocess


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Dict[str, int]] = defaultdict(dict)  # term -> {docid: tf}
        self.df: Dict[str, int] = defaultdict(int)                 # term -> document frequency
        self.doc_tokens: Dict[str, List[str]] = {}                 # docid -> ordered token list
        self.doc_titles: Dict[str, str] = {}                       # docid -> title (for display)
        self.doc_categories: Dict[str, str] = {}                   # docid -> category
        self.N: int = 0                                            # total number of documents

    def build(self, documents: List[Dict[str, str]]):
        """Build the inverted index from a list of parsed documents."""
        self.N = len(documents)

        for doc in documents:
            docid = doc["docid"]
            tokens = preprocess(doc["content"])
            self.doc_tokens[docid] = tokens
            self.doc_titles[docid] = doc["title"]
            self.doc_categories[docid] = doc["category"]

            tf_counts: Dict[str, int] = defaultdict(int)
            for tok in tokens:
                tf_counts[tok] += 1

            for term, tf in tf_counts.items():
                self.index[term][docid] = tf

        # Document frequency = number of documents in which the term appears
        for term, postings in self.index.items():
            self.df[term] = len(postings)

    def get_postings(self, term: str) -> Dict[str, int]:
        """Return {docid: tf} for a given (already-stemmed) term."""
        return self.index.get(term, {})

    def get_df(self, term: str) -> int:
        return self.df.get(term, 0)

    def vocabulary(self):
        return sorted(self.index.keys())

    def save_dictionary_report(self, path: str, top_n_postings: int = 10):
        """
        Write a human-readable dictionary/inverted-index report to disk:
        term, df, and (a sample of) postings (docID, tf).
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"INVERTED INDEX (Part A/B)\n")
            f.write(f"Total documents (N): {self.N}\n")
            f.write(f"Vocabulary size: {len(self.index)}\n")
            f.write("=" * 70 + "\n\n")
            for term in self.vocabulary():
                postings = sorted(self.index[term].items())
                shown = postings[:top_n_postings]
                more = f" ... (+{len(postings) - top_n_postings} more)" if len(postings) > top_n_postings else ""
                f.write(f"{term:20s} df={self.df[term]:3d}  postings={shown}{more}\n")

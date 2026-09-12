"""
vsm.py
------
Part B: Vector Space Model ranked retrieval using lnc.ltc weighting.

Notation (SMART notation lnc.ltc):
    Document vector "lnc":
        l -> log-tf:       wd,t = 1 + log10(tf)          for tf > 0
        n -> no idf
        c -> cosine normalization

    Query vector "ltc":
        l -> log-tf:       (1 + log10(tf_q,t))
        t -> idf:           log10(N / df_t)
        c -> cosine normalization

Cosine similarity is the dot product of the (already normalized)
document and query vectors.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from inverted_index import InvertedIndex
from preprocess import preprocess


class VSMRetriever:
    def __init__(self, inverted_index: InvertedIndex):
        self.idx = inverted_index
        self.N = inverted_index.N
        # Pre-compute normalized document vectors: docid -> {term: weight}
        self.doc_vectors: Dict[str, Dict[str, float]] = {}
        self._build_document_vectors()

    # ------------------------------------------------------------------
    def _build_document_vectors(self):
        """Compute 'lnc' weights for every document and cosine-normalize them."""
        raw_weights: Dict[str, Dict[str, float]] = defaultdict(dict)

        for term in self.idx.vocabulary():
            for docid, tf in self.idx.get_postings(term).items():
                w = 1 + math.log10(tf) if tf > 0 else 0.0
                raw_weights[docid][term] = w

        # Cosine normalization ('c' in lnc)
        for docid, weights in raw_weights.items():
            norm = math.sqrt(sum(w * w for w in weights.values()))
            if norm > 0:
                self.doc_vectors[docid] = {t: w / norm for t, w in weights.items()}
            else:
                self.doc_vectors[docid] = {}

    # ------------------------------------------------------------------
    def _query_vector(self, query: str) -> Dict[str, float]:
        """Compute 'ltc' weights for the query and cosine-normalize."""
        query_terms = preprocess(query)
        tf_counts: Dict[str, int] = defaultdict(int)
        for t in query_terms:
            tf_counts[t] += 1

        raw_weights: Dict[str, float] = {}
        for term, tf in tf_counts.items():
            df = self.idx.get_df(term)
            if df == 0:
                continue  # term not in corpus -> contributes nothing (idf undefined/irrelevant)
            idf = math.log10(self.N / df)
            log_tf = 1 + math.log10(tf)
            raw_weights[term] = log_tf * idf

        norm = math.sqrt(sum(w * w for w in raw_weights.values()))
        if norm > 0:
            return {t: w / norm for t, w in raw_weights.items()}
        return {}

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank documents by cosine similarity to the query using lnc.ltc.
        Returns up to top_k (docid, score) tuples, sorted by decreasing
        score, ties broken by increasing docID.
        """
        q_vec = self._query_vector(query)
        if not q_vec:
            return []

        scores: Dict[str, float] = defaultdict(float)
        for term, q_weight in q_vec.items():
            for docid, d_weight in self._postings_for_term(term):
                scores[docid] += q_weight * d_weight

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k]

    def _postings_for_term(self, term: str):
        """Yield (docid, normalized_doc_weight) pairs for a term."""
        for docid, vec in self.doc_vectors.items():
            if term in vec:
                yield docid, vec[term]

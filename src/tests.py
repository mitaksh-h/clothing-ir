"""
tests.py
--------
Part E: Testing.

Runs:
    - >= 10 free-text queries (VSM)
    - >= 5 exact phrase queries (positional index)
    - >= 3 proximity queries with different k values
    - >= 1 query containing a term absent from the corpus
    - A discussion of >= 2 cases where positional info changes results

All output is written to output/test_report.txt AND printed to stdout.
Also (re)generates output/inverted_index_report.txt and
output/positional_index_report.txt as required deliverables.

No expected document IDs are hard-coded anywhere: every result is
computed live from the corpus at run time.
"""

import io
import os
import sys

from main import build_engine, print_ranked_results, print_phrase_results, print_proximity_results

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


class Tee(io.TextIOBase):
    """Writes to both the terminal and a report file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


FREE_TEXT_QUERIES = [
    "cotton shirt for men",
    "regular fit denim jeans",
    "winter wear jacket",
    "festive kurta for women",
    "breathable fabric t-shirt",
    "stretch leggings",
    "party wear saree",
    "zip closure hoodie",
    "high waist trousers",
    "formal office shirt",
    "unicorn spacesuit accessory",  # contains a term absent from the corpus
]

PHRASE_QUERIES = [
    "cotton shirt",
    "stretch denim",
    "festive wear",
    "winter wear",
    "regular fit",
    "breathable fabric",
]

PROXIMITY_QUERIES = [
    ("cotton", "shirt", 3),
    ("stretch", "denim", 4),
    ("winter", "wear", 3),
    ("festive", "kurta", 4),
]


def run_all_tests(out_stream):
    old_stdout = sys.stdout
    sys.stdout = out_stream
    try:
        documents, inv_index, pos_index, vsm = build_engine()

        print("=" * 70)
        print("PART E: TEST REPORT")
        print("=" * 70)
        print(f"Corpus size (N): {len(documents)}")
        print(f"Vocabulary size: {len(inv_index.vocabulary())}\n")

        print("-" * 70)
        print(f"SECTION 1: FREE-TEXT QUERIES (VSM, lnc.ltc) -- {len(FREE_TEXT_QUERIES)} queries")
        print("-" * 70)
        for q in FREE_TEXT_QUERIES:
            print_ranked_results(vsm, inv_index, q)

        print("\n" + "-" * 70)
        print(f"SECTION 2: EXACT PHRASE QUERIES (positional index) -- {len(PHRASE_QUERIES)} queries")
        print("-" * 70)
        phrase_results = {}
        for p in PHRASE_QUERIES:
            phrase_results[p] = print_phrase_results(pos_index, p)

        print("\n" + "-" * 70)
        print(f"SECTION 3: PROXIMITY QUERIES (positional index) -- {len(PROXIMITY_QUERIES)} queries, varying k")
        print("-" * 70)
        for t1, t2, k in PROXIMITY_QUERIES:
            print_proximity_results(pos_index, t1, t2, k)

        print("\n" + "-" * 70)
        print("SECTION 4: VSM vs POSITIONAL COMPARISON")
        print("-" * 70)
        _compare_vsm_vs_positional(vsm, inv_index, pos_index, phrase_results)

    finally:
        sys.stdout = old_stdout


def _compare_vsm_vs_positional(vsm, inv_index, pos_index, phrase_results):
    """
    Discuss at least two cases where positional info changes the result
    set/order compared to plain VSM 'bag of words' retrieval.
    """
    comparisons = [
        ("cotton shirt", "cotton shirt"),
        ("regular fit", "regular fit"),
        ("winter wear", "winter wear"),
    ]

    for free_text_q, phrase_q in comparisons:
        vsm_results = vsm.search(free_text_q, top_k=10)
        vsm_docids = set(d for d, _ in vsm_results)

        phrase_docs = phrase_results.get(phrase_q)
        if phrase_docs is None:
            phrase_docs = pos_index.phrase_search(phrase_q)
        phrase_docids = set(d for d, _ in phrase_docs)

        print(f"\nQuery terms: {free_text_q!r}")
        print(f"  VSM (bag-of-words) top-10 docIDs   : {sorted(vsm_docids)}")
        print(f"  Positional exact-phrase docIDs     : {sorted(phrase_docids)}")

        only_in_vsm = vsm_docids - phrase_docids
        only_in_phrase = phrase_docids - vsm_docids
        if only_in_vsm:
            print(f"  -> VSM retrieves {sorted(only_in_vsm)} even though the terms do NOT appear "
                  f"as an exact adjacent phrase there (bag-of-words ignores word order/adjacency).")
        if only_in_phrase:
            print(f"  -> Phrase search finds {sorted(only_in_phrase)} which VSM may rank lower/outside "
                  f"top-10 despite the exact phrase being present, since VSM scores by term weight, "
                  f"not adjacency.")
        if not only_in_vsm and not only_in_phrase:
            print("  -> Result sets coincide for this query on this corpus.")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "test_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        tee = Tee(sys.stdout, f)
        run_all_tests(tee)

    print(f"\n[Test report written to {report_path}]")

    # Also regenerate the dictionary / positional-index report deliverables.
    from corpus_parser import load_corpus
    from inverted_index import InvertedIndex
    from positional_index import PositionalIndex

    docs = load_corpus(os.path.join(os.path.dirname(__file__), "..", "corpus", "corpus_100.txt"))

    inv = InvertedIndex()
    inv.build(docs)
    inv.save_dictionary_report(os.path.join(OUTPUT_DIR, "inverted_index_report.txt"))

    pos = PositionalIndex()
    pos.build(docs)
    pos.save_positional_report(os.path.join(OUTPUT_DIR, "positional_index_report.txt"))

    print(f"[Inverted-index report written to {OUTPUT_DIR}/inverted_index_report.txt]")
    print(f"[Positional-index report written to {OUTPUT_DIR}/positional_index_report.txt]")

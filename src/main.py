"""
main.py
-------
Part D: Simple command-line interface for the clothing search engine.

Two search modes are offered:
    1. Free-text ranked search (VSM, lnc.ltc cosine similarity) -> top 10
    2. Phrase / proximity search using the positional index

Run:
    python3 main.py
"""

import os
import sys

from corpus_parser import load_corpus
from inverted_index import InvertedIndex
from positional_index import PositionalIndex
from vsm import VSMRetriever

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "corpus_100.txt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def build_engine():
    documents = load_corpus(CORPUS_PATH)

    inv_index = InvertedIndex()
    inv_index.build(documents)

    pos_index = PositionalIndex()
    pos_index.build(documents)

    vsm = VSMRetriever(inv_index)

    return documents, inv_index, pos_index, vsm


def print_ranked_results(vsm: VSMRetriever, inv_index: InvertedIndex, query: str, top_k: int = 10):
    results = vsm.search(query, top_k=top_k)
    print(f"\nFree-text query: \"{query}\"")
    if not results:
        print("  No matching documents (query terms not found in corpus).")
        return results
    print(f"  {'Rank':<5}{'DocID':<8}{'Score':<10}{'Category':<14}Title")
    for rank, (docid, score) in enumerate(results, start=1):
        title = inv_index.doc_titles.get(docid, "")
        category = inv_index.doc_categories.get(docid, "")
        print(f"  {rank:<5}{docid:<8}{score:<10.4f}{category:<14}{title}")
    return results


def print_phrase_results(pos_index: PositionalIndex, phrase: str):
    results = pos_index.phrase_search(phrase)
    print(f"\nPhrase query: \"{phrase}\"")
    if not results:
        print("  No exact matches found.")
        return results
    print(f"  Found in {len(results)} document(s):")
    for docid, positions in results:
        title = pos_index.doc_titles.get(docid, "")
        print(f"    {docid:<8}{title:<45}start_positions={positions}")
    return results


def print_proximity_results(pos_index: PositionalIndex, term1: str, term2: str, k: int):
    results = pos_index.proximity_search(term1, term2, k)
    print(f"\nProximity query: \"{term1}\" WITHIN/{k} \"{term2}\"")
    if not results:
        print("  No matches found.")
        return results
    print(f"  Found in {len(results)} document(s):")
    for docid, pairs in results:
        title = pos_index.doc_titles.get(docid, "")
        print(f"    {docid:<8}{title:<45}position_pairs={pairs}")
    return results


def interactive_loop():
    print("=" * 70)
    print("  Clothing Search Engine  (CSD358 - Assignment 1)")
    print("=" * 70)
    documents, inv_index, pos_index, vsm = build_engine()
    print(f"Indexed {len(documents)} documents. Vocabulary size: {len(inv_index.vocabulary())}")

    while True:
        print("\nChoose a search mode:")
        print("  1. Free-text ranked search (VSM / cosine similarity)")
        print("  2. Exact phrase search (positional index)")
        print("  3. Proximity search: term1 WITHIN/k term2 (positional index)")
        print("  4. Exit")
        choice = input("Enter choice [1-4]: ").strip()

        if choice == "1":
            query = input("Enter free-text query: ").strip()
            print_ranked_results(vsm, inv_index, query)
        elif choice == "2":
            phrase = input("Enter phrase (e.g. 'cotton shirt'): ").strip()
            print_phrase_results(pos_index, phrase)
        elif choice == "3":
            term1 = input("Enter first term: ").strip()
            term2 = input("Enter second term: ").strip()
            try:
                k = int(input("Enter proximity window k (max tokens between them): ").strip())
            except ValueError:
                print("  Invalid k, defaulting to k=3")
                k = 3
            print_proximity_results(pos_index, term1, term2, k)
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    interactive_loop()

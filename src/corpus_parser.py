"""
corpus_parser.py
----------------
Parses the supplied 100-document clothing corpus (corpus_100.txt).

The corpus file stores each document in a simple pseudo-XML block:

    <DOC>
    <DOCID>D001</DOCID>
    <CATEGORY>T-Shirt</CATEGORY>
    <TITLE>Men's Cotton Crew Neck T-Shirt - Black</TITLE>
    <TEXT>Men's Cotton Crew Neck T-Shirt - Black. Made from 100% cotton ...</TEXT>
    </DOC>

This module reads the raw file and returns a list of dictionaries:
    {"docid": "D001", "category": "T-Shirt", "title": "...", "text": "..."}

We deliberately use simple regular expressions instead of a full XML parser
because the tags are not well-formed XML (e.g. stray '&', '%'), and a
light-weight regex-based reader is more robust for this kind of
"tagged-text" corpus.
"""

import re
from typing import List, Dict


def load_corpus(path: str) -> List[Dict[str, str]]:
    """
    Read the corpus file and return a list of document dictionaries.

    Each dictionary has the keys: docid, category, title, text.
    The 'content' field (title + text concatenated) is what gets fed
    into the indexing pipeline, since a clothing search should match
    against both the title and the description.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split the file into <DOC> ... </DOC> blocks.
    doc_blocks = re.findall(r"<DOC>(.*?)</DOC>", raw, flags=re.DOTALL)

    documents = []
    for block in doc_blocks:
        docid = _extract_tag(block, "DOCID")
        category = _extract_tag(block, "CATEGORY")
        title = _extract_tag(block, "TITLE")
        text = _extract_tag(block, "TEXT")

        if docid is None:
            continue  # skip malformed blocks

        documents.append(
            {
                "docid": docid.strip(),
                "category": (category or "").strip(),
                "title": (title or "").strip(),
                "text": (text or "").strip(),
                # content = what we actually index (title + description)
                "content": f"{(title or '').strip()} {(text or '').strip()}".strip(),
            }
        )

    return documents


def _extract_tag(block: str, tag: str) -> str:
    """Extract the text inside <TAG>...</TAG> from a block of text."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", block, flags=re.DOTALL)
    return match.group(1) if match else None


if __name__ == "__main__":
    docs = load_corpus("../corpus/corpus_100.txt")
    print(f"Loaded {len(docs)} documents.")
    print(docs[0])

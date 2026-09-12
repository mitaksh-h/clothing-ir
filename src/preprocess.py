"""
preprocess.py
-------------
Text pre-processing pipeline shared by both the term-based (Part A/B)
and positional (Part C) indexes, so that phrase/proximity search uses
EXACTLY the same token stream as the VSM index.

Pipeline (applied in this fixed order):
    1. Lower-case the text.
    2. Split on non-alphanumeric characters (punctuation removal).
       Hyphenated / percent forms like "easy-care" or "100%" are split
       into separate tokens ("easy", "care", "100").
    3. Drop pure-numeric tokens (sizes like "100" or "2xl" numeric parts)
       -- kept simple: only tokens that are ENTIRELY digits are dropped,
       since sizes such as S/M/L/XL are already alphabetic and useful,
       while bare numbers ("100" in "100% cotton") carry no retrieval
       value for a clothing search engine.
    4. Remove stop-words using a fixed, documented stop-word list
       (a trimmed version of the standard SMART/NLTK English stop-word
       list -- see STOPWORDS below).
    5. Apply Porter stemming (via NLTK's PorterStemmer, a pure
       rule-based implementation that needs no external downloads).

Stop-word policy (documented, as required by the assignment):
    We use a STANDARD English stop-word list (function words: articles,
    prepositions, pronouns, auxiliary verbs, conjunctions, etc.) and
    apply it uniformly to document text AND to every query, both for
    the VSM (Part B) pipeline and the positional index (Part C)
    pipeline. This keeps semantics of position numbers consistent
    between the two indexes: since stop-words are removed identically
    everywhere, position 3 in the un-stemmed sentence and position 3 in
    the filtered/stemmed token stream always mean the same thing
    relative to each other.

    We do NOT add domain-specific stop-words (e.g. "wear", "fabric")
    even though they are frequent in this corpus, because they still
    carry retrieval-relevant meaning for a clothing search engine
    (e.g. queries like "winter wear" or "breathable fabric" are
    expected positional queries in the assignment brief).
"""

import re
from typing import List
from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()

# Standard English stop-word list (SMART/NLTK-style, ~150 function words).
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over
own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when
when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

# Additional artefact token: apostrophe-s splits ("men's" -> "men", "s")
# produces a bare "s" token with no retrieval value; treat it as a stop-word.
STOPWORDS.add("s")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lower-case and split text into alphanumeric tokens (punctuation removed)."""
    return _TOKEN_RE.findall(text.lower())


def is_pure_number(token: str) -> bool:
    return token.isdigit()


def preprocess(text: str, remove_stopwords: bool = True, stem: bool = True) -> List[str]:
    """
    Full pre-processing pipeline: tokenize -> drop pure numbers ->
    remove stop-words -> stem.

    Returns a list of processed tokens IN ORDER (positions preserved),
    which is required for Part C's positional index.
    """
    tokens = tokenize(text)
    processed = []
    for tok in tokens:
        if is_pure_number(tok):
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        if stem:
            tok = _stemmer.stem(tok)
        processed.append(tok)
    return processed


if __name__ == "__main__":
    sample = "Men's Cotton Crew Neck T-Shirt - Black. Made from 100% cotton, easy-care fabric."
    print(preprocess(sample))

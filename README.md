# Clothing Search Engine — Information Retrieval Assignment 1 (CSD358)

A small search engine over a 100-document clothing product corpus, implementing:

- **Part A** — Tokenization/normalization/stop-word removal/stemming + inverted index (term, df, postings)
- **Part B** — Ranked retrieval using the Vector Space Model, `lnc.ltc` cosine similarity
- **Part C** — Positional index + exact phrase search + ordered proximity (`WITHIN/k`) search
- **Part D** — Interactive command-line application (two search modes)
- **Part E** — Test suite: 11 free-text queries, 6 phrase queries, 4 proximity queries (varying k), one OOV query, and a VSM-vs-positional comparison

## Project structure

```
ir_assignment/
├── corpus/
│   └── corpus_100.txt              # supplied 100-document corpus
├── src/
│   ├── corpus_parser.py            # parses the <DOC>...</DOC> corpus format
│   ├── preprocess.py                # tokenize, stop-words, Porter stemming
│   ├── inverted_index.py           # Part A/B: term -> df -> {docid: tf}
│   ├── positional_index.py         # Part C: term -> df -> {docid: [positions]}
│   ├── vsm.py                       # Part B: lnc.ltc cosine similarity
│   ├── main.py                      # Part D: interactive CLI application
│   ├── tests.py                     # Part E: mandatory test queries + report
│   └── generate_screenshots.py     # renders demo screenshots of the CLI
├── output/
│   ├── inverted_index_report.txt    # dictionary/inverted-index deliverable
│   ├── positional_index_report.txt  # positional-index deliverable
│   ├── test_report.txt              # full Part E test run output
│   └── screenshots/                 # demo screenshots (see note below)
└── README.md
```

## Requirements

- Python 3.8+
- `nltk` (only used for `nltk.stem.PorterStemmer`, a pure rule-based
  algorithm — **no corpus downloads required**)

```bash
pip install nltk
```

## Running

**Interactive application (Part D):**
```bash
cd src
python3 main.py
```
You will be prompted to choose:
1. Free-text ranked search (VSM / cosine similarity) — top 10 results
2. Exact phrase search (positional index)
3. Proximity search — `term1 WITHIN/k term2` (positional index)
4. Exit

**Regenerate all reports + run the full Part E test suite:**
```bash
cd src
python3 tests.py
```
This (re)writes `output/test_report.txt`, `output/inverted_index_report.txt`,
and `output/positional_index_report.txt`.

**Regenerate demo screenshots:**
```bash
cd src
python3 generate_screenshots.py
```

## Design decisions

### Pre-processing pipeline (Part A)
1. Lower-case the text.
2. Tokenize on non-alphanumeric characters — this both strips
   punctuation and splits hyphenated/compound tokens
   (e.g. `"t-shirt"` → `"t"`, `"shirt"`; `"easy-care"` → `"easy"`, `"care"`).
3. Drop tokens that are purely numeric (e.g. the `"100"` in `"100% cotton"`),
   since bare numbers carry no useful retrieval signal for this corpus.
   Size tokens like `S`, `M`, `L`, `XL`, `2XL` are alphanumeric mixes or
   pure letters and are kept.
4. Remove stop-words using a **standard English function-word list**
   (articles, prepositions, pronouns, auxiliary verbs, conjunctions —
   see `STOPWORDS` in `preprocess.py`), applied identically to
   documents and to every query, in both the VSM pipeline and the
   positional-index pipeline. One project-specific addition: the
   token `"s"` (an artifact of splitting possessives like `"men's"`)
   is also treated as a stop-word.
   We deliberately do **not** add domain words such as `"wear"` or
   `"fabric"` to the stop-list, even though they're frequent, because
   they carry real query meaning here (e.g. `"winter wear"`,
   `"breathable fabric"` are expected phrase queries).
5. Apply Porter stemming (`nltk.stem.PorterStemmer`).

Because both the term-based index (Part A/B) and the positional index
(Part C) run the *exact same* `preprocess()` function, token positions
are consistent across both indexes, and phrase/proximity results are
directly comparable to VSM rankings.

### Vector Space Model (Part B) — `lnc.ltc`
- **Document weight:** `w_d,t = 1 + log10(tf)` (no idf), cosine-normalized.
- **Query weight:** `w_q,t = (1 + log10(tf_q,t)) * log10(N / df_t)`, cosine-normalized.
- **Score:** dot product of normalized document and query vectors.
- Query terms with `df = 0` (not in the corpus) are simply ignored —
  this is how the "term not in corpus" test case is handled: if
  *every* query term is OOV the result set is empty and the CLI
  reports "No matching documents".
- Ties are broken by increasing docID (Python's stable sort combined
  with a `(-score, docid)` sort key).

### Positional index (Part C)
- Structure: `term -> df -> {docid: [p1, p2, ...]}`, 0-based positions
  into the pre-processed token stream.
- **Phrase search**: for a k-term phrase, take the position list of
  the first term as candidate anchor positions, then require every
  subsequent query term to appear at `anchor + offset` in the *same*
  document — i.e., true adjacency, not just co-occurrence.
- **Proximity search** (`term1 WITHIN/k term2`): ordered — term1 must
  occur at position `p1`, term2 at position `p2`, with
  `0 < p2 - p1 <= k + 1` (at most `k` tokens strictly between them).

### VSM vs. positional retrieval (Part E discussion)
Section 4 of `output/test_report.txt` compares, for three query pairs
(`"cotton shirt"`, `"regular fit"`, `"winter wear"`), the top-10 VSM
docIDs against the positional exact-phrase docIDs. In every case the
two sets differ:
- VSM (bag-of-words) also ranks documents where both terms occur
  *without* being adjacent (e.g. "...men's cotton crew neck t-shirt...")
  highly, purely because both stems are present and weighted — it has
  no notion of word order.
- The positional/phrase index only returns documents where the
  words are truly adjacent (e.g. "...checked cotton shirt..."), and
  can surface documents (e.g. for "regular fit") that fall outside
  the VSM top-10 window entirely, because VSM ranks by *term weight*
  across the whole query, not by adjacency of any specific pair.

This directly demonstrates why positional information changes both
the **result set** and the **implied relevance** compared to plain
VSM retrieval, satisfying the Part E requirement.

## Deliverables checklist
- [x] Source code with comments (`src/`)
- [x] Dictionary/inverted-index output (`output/inverted_index_report.txt`)
- [x] Positional-index output (`output/positional_index_report.txt`)
- [x] Screenshots of the application (`output/screenshots/`) — these are
      auto-generated captures of real program output; you may replace
      them with your own OS-level terminal screenshots before submission.
- [x] Full test report, 11 free-text + 6 phrase + 4 proximity queries,
      one OOV query, VSM-vs-positional discussion (`output/test_report.txt`)


- Mitaksh Goswami 2410110875


*This project was created as part of the 42 curriculum by samarkar.*

# RAG Against The Machine

## Description

A Retrieval-Augmented Generation system that answers questions about the vLLM
codebase.

A language model only knows what it was trained on. Rather than retraining it,
this system lets it reach for the right information at answering time: the
question is matched against an index built from the corpus, the most relevant
snippets are retrieved, and a small local model (Qwen/Qwen3-0.6B) writes an
answer grounded in them.

Four stages: **indexing** (chunk the files, build the index), **retrieval**
(rank chunks against a question), **augmentation** (fit the best ones into the
model's context), **generation** (produce the answer).

## Instructions

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
make install          # uv sync
make lint             # flake8 + mypy
make clean            # remove __pycache__ and .mypy_cache
make run              # show the CLI help
make debug            # run under pdb
```

The corpus and the datasets are not shipped with this repository. Place them
as follows before running anything:

```
data/raw/vllm-0.10.1/                 the sources to index
data/datasets/UnansweredQuestions/    questions to search
data/datasets/AnsweredQuestions/      ground truth, for evaluation
```

`index` writes to `data/processed/`; the dataset commands write wherever
`--save_directory` points.

## System Architecture

```
                    ┌──────────────┐
   data/raw/  ────► │ CodeIndexer  │ ────►  data/processed/
                    └──────────────┘         ├── chunks.json
                            │                └── bm25_vectorizer.pkl
                            ▼
                    ┌──────────────┐
                    │  tokenize()  │  ◄── shared with the retriever
                    └──────────────┘
                            ▲
                            │
   question ────►   ┌──────────────┐
                    │  Retriever   │ ────►  top-k chunks
                    └──────────────┘             │
                                                 ▼
                                         ┌──────────────┐
                                         │  Generator   │ ────►  answer
                                         └──────────────┘
```

| Module | Responsibility |
|---|---|
| `src/indexer.py` | chunks files, builds and serialises the BM25 index |
| `src/tokenizer.py` | the single tokenisation function, shared by indexing and querying |
| `src/retriever.py` | loads the index, ranks chunks against a query |
| `src/generator.py` | builds the prompt, runs the model, returns the answer |
| `src/models.py` | the eight pydantic models exchanged between stages |
| `src/__main__.py` | the Fire CLI wiring it together |

Indexing produces two artefacts with distinct roles. `bm25_vectorizer.pkl`
holds the BM25 statistics and answers *which chunks match, in what order*;
`chunks.json` holds the records and answers *what does chunk n contain, and
where does it live*. The retriever needs both.

`tokenize()` lives in its own module because it runs twice — once per chunk at
indexing time, once per query. If the two ever diverged, BM25 would compare
tokens that no longer resemble each other, and recall would collapse silently.

## Chunking Strategy

Two strategies, one per file kind, converging on the same machinery.

- **Python** — cut on top-level `class` and `def`, so a chunk holds a complete
  definition.
- **Markdown and text** — cut on headings, so a chunk holds a section.

A section over `--max_chunk_size` is re-cut on blank lines, and sliced at a
fixed width as a last resort. That fallback matters: the grader rejects any
source longer than 2000 characters, and one oversized source invalidates the
whole submission. Every chunk passes through `_add_chunk`, the only place that
enforces the limit.

**Positions, not lengths.** The chunker never rebuilds text. Boundaries come
from `re.finditer()`, content from slicing the original string. Because the
content is derived from the indices, the two cannot contradict each other —
all 18 792 chunks verify `source_file[first:last] == content` exactly. This was
learned the hard way (see *Challenges*).

## Retrieval Method

**BM25** (Okapi BM25, via `rank_bm25`). For each query term it combines term
frequency with saturating returns — a term repeated twenty times does not score
twenty times higher — and inverse document frequency, so `KVCacheManager`
weighs far more than `the`. Scores are normalised by chunk length, so a long
chunk cannot win on volume alone.

**Tokenisation** is where most of the recall was won. Source code does not read
like prose: identifiers are compound, punctuation is glued to names, and a
question may either quote an identifier or paraphrase it. `tokenize()` drops
punctuation and emits a compound identifier **whole and in parts**:

```
"fused_batched_moe"  ->  ['fused_batched_moe', 'fused', 'batched', 'moe']
"getNumTokens"       ->  ['getnumtokens', 'get', 'num', 'tokens']
"serve()"            ->  ['serve']
```

A question quoting `fused_batched_moe` matches the whole token; one asking
about "the fused batched MoE layer" matches the parts.

## Performance Analysis

Measured on the public datasets (100 questions each), CPU only.

| | @1 | @3 | **@5** | @10 | required @5 |
|---|---|---|---|---|---|
| **docs** | 0.610 | 0.750 | **0.820** | 0.850 | 0.80 |
| **code** | 0.414 | 0.646 | **0.677** | 0.768 | 0.50 |

| Operation | Measured | Limit |
|---|---|---|
| Indexing 1 971 files → 18 792 chunks | 4.5 s | 300 s |
| Searching 100 questions | ~6 s | 90 s / 200 questions |
| Generating one answer | 37.3 s | none stated |

Generation is slow in reading, not writing. Answers average 291 characters —
about 75 tokens, well under the 256-token cap — while the prompt carries five
chunks of up to 2000 characters each. Running `answer --k 0`, with an empty
context, takes 7 s instead of 37 s, which confirms where the time goes.

### What moved recall

Two structural changes account for nearly all of it.

**Tokenisation**, which improved code recall fourfold:

| Tokeniser | docs@5 | code@5 |
|---|---|---|
| `text.lower().split()` | 0.730 | 0.162 |
| split on non-alphanumeric | 0.790 | 0.556 |
| **+ camelCase splitting** | **0.790** | **0.677** |

**File coverage.** Three percent of reference sources lived in `.txt` files
that were never indexed, capping docs recall at 0.970 regardless of ranking:

| Indexed | ceiling | docs@5 | code@5 |
|---|---|---|---|
| `.md` + `.py` | 0.970 | 0.790 | 0.677 |
| **+ `.txt`** | **1.000** | **0.820** | **0.677** |

### What did not

**Chunk size.** Sweeping `--max_chunk_size` from 400 to 2000 showed no clear
trend. Imposing a minimum chunk size was also measured:

| minimum size | docs@5 | code@5 |
|---|---|---|
| **none (current)** | **0.820** | **0.677** |
| 100 | 0.820 | 0.677 |
| 200 | 0.830 | 0.616 |
| 600 | 0.770 | 0.636 |

So: *too small* has no measurable effect below a couple of hundred characters —
a chunk holding nothing but a heading never wins a ranking and never disturbs
one. Merging harder does hurt, losing semantic boundaries. *Too large*, and the
2000-character hard limit bites first.

**BM25 parameters.** Sweeping `k1` and `b` moved results by at most 0.05,
indistinguishable from noise on 100 questions. One combination landed docs@5 at
exactly 0.800 — precisely the threshold — and was rejected: hitting a bar
exactly on the public set, by tuning a parameter with no principled
justification, is how a system passes locally and fails on held-out data.

## Design Decisions

**BM25 over TF-IDF.** BM25 adds saturating term frequency and length
normalisation, both of which matter when chunks range from a heading to 2000
characters.

**One tokeniser, one module.** Indexing and querying must agree exactly;
sharing a function makes divergence impossible rather than merely unlikely.

**Positions derived from the source, never accumulated.** Indices and content
cannot disagree when one is sliced from the other.

**`k` and the generation budget are separate.** Retrieval returns `k` sources —
10 by default, which is what gets scored — but only the first five reach the
model. All ten would mean up to 20 000 characters of context for a 0.6B model:
slower, and more diluted.

**`answer_dataset` re-reads the source files rather than re-running search.**
The results file is a contract: *these are the sources I stand behind*.
Re-searching could return different chunks if the index changed, producing an
answer grounded in passages the output does not list.

**`rank_bm25` over `bm25s`.** `bm25s` is newer and faster, but indexing uses
1.5% of its time budget and retrieval 8% of its own — it would solve a problem
this project does not have, at the cost of a new serialisation format and a
full re-validation. It would be the right call one order of magnitude up.

**Chat template for generation.** The first working version produced the
correct answer, then repeated it twenty times until the token budget ran out.
Qwen3 expects a conversation format; fed raw text it has no signal for where to
stop. `apply_chat_template()` supplies the end-of-turn token.

## Challenges Encountered

### Character offsets that quietly drifted

The first chunker computed positions by accumulating the lengths of
reconstructed text. Sections were split on `\n\n+` — two newlines *or more* —
and rejoined with exactly `"\n\n"`, so a three-newline separator became two:
the text was altered, its length with it, and the error accumulated down the
file through a running counter.

Measured over 18 319 chunks: 14 804 offsets exact (81%), and **2 774 (18%)
whose stored content appeared nowhere in the source file**.

This is invisible in normal use — the retrieved text still looks right — but
the grader counts a source as found only if its character range overlaps the
reference. Perfect retrieval with drifted offsets scores zero.

The fix was to stop deriving positions from lengths. After the rewrite:
**18 792 chunks, 100% exact, none over 2000 characters.**

### Code that does not read like prose

Code recall started at **0.162** against a 0.50 requirement, while docs recall
was already near target. The cause was tokenisation, above. The lesson
generalises: the index and the question have to meet somewhere, and for a
codebase that place is neither plain words nor raw identifiers, but both.

### A retrieval ceiling

Docs recall sat at 0.790 against a 0.80 requirement and no ranking work moved
it. Measuring the *ceiling* — what fraction of reference sources were even
reachable given what had been indexed — showed 0.970. Three percent of the
answers lived in `.txt` files the indexer never opened. That question, *is my
ranking bad or is my corpus incomplete*, cannot be answered by recall alone.

### A case lexical matching cannot solve

For *"What method does vLLM's LLM class provide for generating embedding
vectors from prompts?"*, the correct chunk (`### LLM.embed`) sits in the index
with an IoU of 0.40 against the reference, but BM25 ranked it outside the top
ten and returned the adjacent `### LLM.score` instead. The question says
*"generating"*, and `generate` is among the most frequent terms in the corpus.
No boundary adjustment recovers this — it is the case a semantic index would
handle.

## Example Usage

```bash
# Index the corpus
uv run python -m src index --max_chunk_size 2000
# Ingestion complete! Indexes are saved under data/processed/

# One question
uv run python -m src search "How to configure OpenAI server?" --k 10
uv run python -m src answer "How to install vLLM?" --k 10

# A whole dataset — scope --save_directory per dataset, the public
# datasets share file names
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions

# Score your own retrieval
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
# Recall@1: 0.610  Recall@3: 0.750  Recall@5: 0.820  Recall@10: 0.850
```

Degenerate inputs exit with a message rather than a traceback:

```bash
uv run python -m src search "" --k 10
uv run python -m src search "asdfghjkl" --k 10
uv run python -m src answer "What is vLLM?" --k 0
uv run python -m src search_dataset --dataset_path /nonexistent.json --save_directory /tmp/out
```

## Resources

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009) — BM25 and how it differs from TF-IDF
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020) — the paper that named the approach
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) — the implementation
  used, and [bm25s](https://github.com/xhluca/bm25s), the faster alternative
  evaluated above
- [vLLM documentation](https://docs.vllm.ai/) — the corpus itself
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) and
  [transformers](https://huggingface.co/docs/transformers/) — chat template and
  generation
- [pydantic](https://docs.pydantic.dev/),
  [Python Fire](https://github.com/google/python-fire)

### Use of AI

- **Concepts** — BM25 scoring, TF and IDF, pydantic inheritance, worked
  through with numeric examples.
- **Debugging** — the character-offset drift was diagnosed this way.
- **Code** - README

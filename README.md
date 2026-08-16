*This project was created as part of the 42 curriculum by samarkar.*

# RAG Against The Machine

## Description

A Retrieval-Augmented Generation system that answers questions about the vLLM
codebase.

A language model only knows what it was trained on. Rather than retraining it,
this system gives it access to an external source at answering time: the
question is matched against an index built from the corpus, the most relevant
snippets are retrieved, and a small local model (Qwen/Qwen3-0.6B) writes an
answer grounded in them.

The pipeline has four stages:

1. **Indexing** — split every source file into chunks and build a BM25 index
2. **Retrieval** — rank the chunks against a question, return the top-k
3. **Augmentation** — fit the selected snippets into the model's context
4. **Generation** — produce an answer grounded in that context

The system is judged on how well it retrieves the right source locations, and
on whether its answers are grounded in them.

## Instructions

### Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) as project and package manager

### Install

```bash
make install          # runs: uv sync
```

### Corpus layout

The corpus and the question datasets are not shipped with this repository.
Place them as follows before running anything:

```
data/raw/vllm-0.10.1/                      the vLLM sources to index
data/datasets/UnansweredQuestions/         questions to search
data/datasets/AnsweredQuestions/           ground truth, for evaluation
```

The `index` command writes to `data/processed/`, and the dataset commands
write wherever `--save_directory` points.

### Development

```bash
make lint             # flake8 + mypy
make clean            # remove __pycache__ and .mypy_cache
make run              # show the CLI help
make debug            # run under pdb
```

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
| `src/indexer.py` | splits files into chunks, builds and serialises the BM25 index |
| `src/tokenizer.py` | the single tokenisation function, used by both indexing and querying |
| `src/retriever.py` | loads the index, ranks chunks against a query |
| `src/generator.py` | builds the prompt, runs Qwen/Qwen3-0.6B, returns the answer |
| `src/models.py` | the eight pydantic models exchanged between stages |
| `src/__main__.py` | the Fire CLI wiring all of the above |

Two artefacts are produced by indexing and they play different roles.
`bm25_vectorizer.pkl` holds the BM25 statistics — term frequencies, inverse
document frequencies, document lengths — and answers *which chunks match, and
in what order*. `chunks.json` holds the chunk records and answers *what does
chunk n contain, and where does it live in the corpus*. The retriever needs
both: BM25 ranks, `chunks.json` supplies the payload.

`tokenize()` living in its own module is deliberate. It is called once at
indexing time and once per query; if the two ever diverged, BM25 would compare
tokens that no longer resemble each other and recall would collapse silently,
with no error raised anywhere.

## Chunking Strategy

Two distinct strategies, one per file kind, both converging on the same
machinery.

**Python** — cut on top-level `class` and `def` boundaries, so a chunk holds a
complete definition rather than an arbitrary window.

**Markdown and text** — cut on headings (`#` through `######`), so a chunk holds
a section.

A section longer than `--max_chunk_size` is then re-cut on blank-line
boundaries, and as a last resort sliced at a fixed width. That last fallback is
not decorative: the scorer rejects any retrieved source longer than 2000
characters, and a single oversized source invalidates the whole submission.
Every chunk goes through one function, `_add_chunk`, which is the only place
that enforces the limit.

### Positions, not lengths

The chunker never rebuilds text. Boundaries are found with `re.finditer()`,
which reports where each match starts, and the content is obtained by slicing
the original string:

```python
content = text[start:end]
```

This is the single most important decision in the indexer, and it was learned
the hard way (see *Challenges*). Because the content is derived from the
indices, the two cannot contradict each other. All 18 865 chunks verify
`source_file[first:last] == content` exactly.

## Retrieval Method

**BM25** (Okapi BM25, via `rank_bm25`), a lexical ranking function.

For a query term, BM25 combines two quantities: how often the term appears in a
chunk (term frequency, with saturating returns so a term repeated twenty times
does not score twenty times higher), and how rare the term is across the whole
corpus (inverse document frequency, so `KVCacheManager` weighs far more than
`the`). Scores are normalised by chunk length, so a long chunk does not win
simply by containing more words.

Retrieval loads the pickled index once, tokenises the query with the same
function used at indexing time, and returns the top-k chunks ranked by score.

### Tokenisation

This is where most of the recall was won. Source code does not read like prose:
identifiers are compound (`fused_batched_moe`, `getNumTokens`), punctuation is
glued to names (`serve()`, `vllm.entrypoints.openai`), and a question may
either quote an identifier literally or paraphrase it in plain words.

`tokenize()` handles both. Punctuation is dropped, and a compound identifier is
emitted **whole and in parts**:

```
"fused_batched_moe"  ->  ['fused_batched_moe', 'fused', 'batched', 'moe']
"getNumTokens"       ->  ['getnumtokens', 'get', 'num', 'tokens']
"serve()"            ->  ['serve']
```

A question quoting `fused_batched_moe` matches on the whole token; a question
asking about "the fused batched MoE layer" matches on the parts.

## Performance Analysis

All figures measured on the public datasets (100 questions each), on an
Apple Silicon laptop, CPU only.

### Recall

| | @1 | @3 | **@5** | @10 | required @5 |
|---|---|---|---|---|---|
| **docs** | 0.610 | 0.750 | **0.820** | 0.850 | 0.80 |
| **code** | 0.414 | 0.646 | **0.677** | 0.768 | 0.50 |

### Throughput

| Operation | Measured | Limit |
|---|---|---|
| Indexing 1 943 files → 18 865 chunks | 4.5 s | 300 s |
| Searching 100 questions (index load included) | ~6 s | 90 s / 200 questions |
| Generating one answer | 37.3 s | none stated |

Generation is the slow stage, and the cost is in reading rather than writing.
Answers average 291 characters, roughly 75 tokens — well under the 256-token
cap, so the model stops on its own. The prompt, however, carries five chunks of
up to 2000 characters each: the model spends about twenty times longer
ingesting the context than producing the answer. Running `answer --k 0`, with
an empty context, takes 7 s instead of 37 s, which confirms where the time
goes.

### What actually moved recall

Two changes account for nearly all of it, and both are structural rather than
tuned to the public data.

**Tokenisation:**

| Tokeniser | docs@5 | code@5 |
|---|---|---|
| `text.lower().split()` | 0.730 | 0.162 |
| split on non-alphanumeric | 0.790 | 0.556 |
| **+ camelCase splitting** | **0.790** | **0.677** |

Code recall improved fourfold. With naïve splitting, a question ending in
`vLLM?` produced the token `vllm?`, which matches nothing, and
`fused_batched_moe` stayed a single opaque token.

**File coverage:**

| Indexed extensions | ceiling | docs@5 | code@5 |
|---|---|---|---|
| `.md` + `.py` | 0.970 | 0.790 | 0.677 |
| **`.md` + `.py` + `.txt`** | **1.000** | **0.820** | **0.677** |

Three percent of the ground-truth sources lived in `.txt` files that were never
indexed, capping docs recall at 0.970 no matter how good the ranking was.
Adding 163 chunks lifted the ceiling to 1.000 and docs@5 past the threshold.

### What did not move recall

**Chunk size.** Sweeping `--max_chunk_size` from 400 to 2000 produced no clear
trend — the curve is noisy, within a few points either way. Merging small
sections was also measured, by imposing a minimum chunk size:

| minimum size | docs@5 | code@5 |
|---|---|---|
| **none (current)** | **0.820** | **0.677** |
| 100 | 0.820 | 0.677 |
| 200 | 0.830 | 0.616 |
| 600 | 0.770 | 0.636 |

Below 100 characters the effect is nil: degenerate chunks holding nothing but a
heading are inert — they never win a ranking, and they do not disturb one
either. Merging harder does hurt: at 600, semantic boundaries are lost and both
datasets drop.

So the honest answer to *what happens if chunks are too small* is: below a
couple of hundred characters, nothing measurable. *Too large*, and the hard
constraint bites first — the scorer rejects sources over 2000 characters.

**BM25 parameters.** Sweeping `k1` and `b` moved results by at most 0.05 in
either direction, indistinguishable from noise on a 100-question sample. One
combination (`k1=2.2, b=0.75`) landed docs@5 at exactly 0.800 — precisely the
threshold. It was rejected: hitting a bar exactly, on the public set, by tuning
a parameter with no principled justification is how a system passes locally and
fails on held-out data. The defaults are kept.

## Design Decisions

**BM25 over TF-IDF.** Both are allowed. BM25 adds two things TF-IDF lacks:
saturating term frequency, so repetition stops paying after a point, and length
normalisation, so long chunks do not win by volume alone. Both matter here,
where chunk sizes range from a heading to 2000 characters.

**One tokeniser, one module.** Indexing and querying must agree exactly. Sharing
a function makes divergence impossible rather than merely unlikely.

**Character positions derived from the source, never accumulated.** Content is
sliced out of the original text, so indices and content cannot disagree.

**`k` and the generation budget are separate.** Retrieval returns `k` sources
(10 by default, which is what gets scored), but only the first five reach the
model. Feeding all ten would mean up to 20 000 characters of context for a
0.6B model — slower, and more diluted. Retrieving widely and reading narrowly
serves both goals at once.

**`answer_dataset` re-reads the source files rather than re-running search.**
The search results file is a contract: *these are the sources I stand behind*.
Re-searching could return different chunks if the index changed between runs,
producing an answer grounded in passages the output file does not list. Re-read
is also why exact character offsets matter twice over.

**`rank_bm25` over `bm25s`.** `bm25s` is newer and considerably faster, using
sparse matrices. It was not adopted: indexing uses 1.5% of its time budget and
retrieval 8% of its own, so the speed it buys solves a problem this project
does not have, while costing a rewrite of the serialisation format and a full
re-validation of recall. It would be the right call if the corpus grew by an
order of magnitude.

**Chat template for generation.** The first working version produced the correct
answer and then repeated it twenty times until the token budget ran out. Qwen3
is an instruction-tuned model expecting a conversation format; feeding it raw
text left it with no signal for where to stop. Wrapping the prompt with
`apply_chat_template()` gives the model its end-of-turn token, and generation
terminates on its own.

## Challenges Encountered

### Character offsets that quietly drifted

The first chunker computed positions by accumulating the lengths of
reconstructed text. Sections were split on `\n\n+` — two newlines *or more* —
and rejoined with exactly `"\n\n"`. A three-newline separator therefore became
two: the text was altered, its length with it, and since the next chunk's
position was that length added to a running counter, the error accumulated over
the whole file. Markdown was worst affected, because the counter was carried
across sections; and the text preceding the first heading was dropped entirely,
offsetting everything after it.

Measured over 18 319 chunks: 14 804 offsets exact (81%), and **2 774 (18%)
whose stored content did not appear anywhere in the source file**.

This is invisible in normal use — the retrieved text still looks right — but
the scorer counts a source as found only if its character range overlaps the
reference. Perfect retrieval with drifted offsets scores zero.

The fix was to stop deriving positions from lengths. Boundaries now come from
`re.finditer().start()`, content from slicing the original string. Verification
after the rewrite: **18 865 chunks, 100% exact, none exceeding 2000
characters.**

### Code that does not read like prose

Initial code recall was **0.162** against a 0.50 requirement, while docs recall
was already near target. The cause was tokenisation, described above. The
lesson generalises: the index and the question have to meet somewhere, and for
a codebase that place is neither plain words nor raw identifiers, but both.

### A retrieval ceiling nobody had noticed

Docs recall sat at 0.790 against a 0.80 requirement, and no amount of ranking
work moved it. Measuring the ceiling — what fraction of ground-truth sources
were even reachable given what had been indexed — showed 0.970. Three percent
of the answers lived in `.txt` files the indexer never opened. No ranking
improvement could ever have found them.

### A case the lexical index cannot solve

For the question *"What method does vLLM's LLM class provide for generating
embedding vectors from prompts?"*, the correct chunk (`### LLM.embed`) exists in
the index with an IoU of 0.40 against the reference — but BM25 ranked it
outside the top ten, returning the adjacent `### LLM.score` chunk at rank 8
instead. The question says *"generating embedding vectors"*, and `generate` is
one of the most frequent terms in the entire vLLM corpus, so it pulls toward
the wrong sections.

This is a limitation of lexical matching, not of chunking: no boundary
adjustment recovers it. It is the case a semantic index would handle, and it is
the main reason semantic embeddings top the list of what to do next.

## Example Usage

### Index the corpus

```bash
uv run python -m src index --max_chunk_size 2000
```

```
Indexing: 100%|██████████████████████████| 1943/1943 [00:00<00:00, 12436.85it/s]
Ingestion complete! Indexes are saved under data/processed/
```

### Search a single question

```bash
uv run python -m src search "How to configure OpenAI server?" --k 10
```

Returns a `MinimalSearchResults`: the question, a generated id, and ten source
locations.

### Search a whole dataset

Always scope `--save_directory` per dataset — the public datasets share file
names, so writing every run to the same folder would overwrite earlier results.

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

```
Searching: 100%|████████████████████████████| 100/100 [00:03<00:00, 30.65it/s]
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json
```

### Answer a single question

```bash
uv run python -m src answer "How to install vLLM?" --k 10
```

```
answer='To install vLLM, follow these steps based on the provided context:

1. **Use pip**: If you are using NVIDIA GPUs, you can install vLLM directly
   via pip.

   ```bash
   uv pip install vllm --torch-backend=auto
   ```
...'
```

### Answer a whole dataset

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Expect roughly one hour for 100 questions on CPU.

### Score your own retrieval

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

```
Recall@1: 0.610  Recall@3: 0.750  Recall@5: 0.820  Recall@10: 0.850
```

### Degenerate inputs

All of these exit cleanly, with a message rather than a traceback:

```bash
uv run python -m src search "" --k 10
uv run python -m src search "asdfghjkl" --k 10
uv run python -m src answer "What is vLLM?" --k 0
uv run python -m src search_dataset --dataset_path /nonexistent.json --save_directory /tmp/out
```

## Resources

### Retrieval

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009) — the reference on BM25 and how it differs from TF-IDF
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) — the implementation used
- [bm25s](https://github.com/xhluca/bm25s) — a faster alternative, evaluated
  and documented above

### Retrieval-Augmented Generation

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP
  Tasks* (2020) — the paper that named the approach

### Tooling

- [vLLM documentation](https://docs.vllm.ai/) — the corpus itself
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — chat template and
  generation parameters
- [transformers documentation](https://huggingface.co/docs/transformers/) —
  `apply_chat_template`, `generate`
- [pydantic](https://docs.pydantic.dev/) — validation and serialisation
- [Python Fire](https://github.com/google/python-fire) — the CLI

### Use of AI

- **Concepts** — BM25 scoring, TF and IDF, pydantic inheritance, worked
  through with numeric examples.
- **Debugging** — the character-offset drift was diagnosed this way.
- **Code** - README

# Reusing the Hugging Face BPE benchmark for a Python Rewrite baseline

## Read-only inventory

The relevant project is `/root/code/tokenizers-bpe-benchmark/`. It has no `AGENTS.md`; `/root/code/tokenizers` also has no `AGENTS.md` found. The benchmark README is the detailed protocol: `/root/code/tokenizers-bpe-benchmark/README.md`. Main files are `run_bench.py` (case matrix, subprocess runner, JSONL), `python_bench.py` (HF Python binding end-to-end runner), `prepare_corpus.py` (deterministic corpus selection), `download_corpora.py` (pinned source download/checksum), `corpus_stats.py`, `instrument_train.py`, `summarize.py`, and `compare_models.py`.

No benchmark was run. No source or benchmark files were changed.

## Data already on disk

The benchmark's `data/raw` and `data/text` are symlinks to `/tmp/tokenizers-bpe-bench/data/raw` and `/tmp/tokenizers-bpe-bench/data/text`. Both targets exist and are populated. I confirmed paths and file sizes only; I did not read the corpus text. Four downloaded raw Parquet shards are available:

| Language | Raw shard size | Prepared line samples available |
|---|---:|---|
| English (`en`) | 188 MB | 1, 4, 16, 32 MiB |
| Chinese (`zh`) | 127 MB | 1, 4, 16, 32 MiB |
| German (`de`) | 220 MB | 1, 4, 16, 32 MiB |
| Japanese (`ja`) | 176 MB | 1, 4, 16, 32 MiB |

The source is Wikimedia Wikipedia's 2023-11-01 snapshot, dataset revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Each language has a pinned Parquet shard, SHA-256, and prepared sample hashes in `data/text/{en,zh,de,ja}-manifest.json`. Samples are made from paragraphs after collapsing whitespace, retaining rows 32–8192 UTF-8 bytes, and selecting deterministically by BLAKE2b-64(article ID, paragraph index). Sizes are byte-matched approximately across languages; samples at larger sizes extend the same deterministic order. README records the dataset's CC BY-SA 3.0 / GFDL terms. The actual data can be read directly from the existing `/tmp/tokenizers-bpe-bench/data/text` path without downloading or preparing it again.

The benchmark also already contains results JSONL and a summary from prior HF runs. These are useful for understanding the harness and reference ranges, but are not measurements of the user's Python trainer.

## Existing benchmark shape and useful conventions

`run_bench.py` runs one case in a child process, captures JSON output and stage timings, writes JSONL, shuffles jobs with a fixed seed, supports repeats/timeouts, and skips successful cases already recorded. Relevant profiles include `quick`, `preprocess`, `bytelevel`, `large_preprocess`, `scaling`, `parallel`, and `full`. The language/size/split/mode/vocabulary are recorded per case.

The Python runner at `python_bench.py` is specifically HF `tokenizers==0.23.2`, not a baseline for `efficient_bpe`. It tests `Tokenizer(models.BPE())` plus optional `WhitespaceSplit`, `Whitespace`, or `ByteLevel` pretokenization and calls either `tokenizer.train([file])` or `train_from_iterator(lines(file))`. Its output is an effective schema to copy for a separate Rewrite runner: engine/version, input bytes, split/mode, requested and actual vocabulary and merges, elapsed milliseconds, peak RSS, and model digest. It uses `min_frequency=2` and usually vocab 8,000 for 1/4 MiB and 16,000 for 16/32 MiB. The repo's Python runner hashes vocabulary in ID order and the ordered merge list, so outputs can be checked beyond vocab size alone.

`run_bench.py`'s `stages` mode is for instrumented Rust source and is not directly reusable for Python. A Python Rewrite process can use the same case parameters and JSONL envelope, with wall time and peak RSS. If useful, add Python phase times around its existing training stages (initial corpus/pair count, merge loop, output) without changing the algorithm. Keep elapsed end-to-end time separate from phase timing.

## Semantic alignment cautions for the Rewrite comparison

1. Freeze the exact input bytes, language, sample size, vocabulary target, and minimum pair frequency. Keep the same line/newline treatment on both sides. The prepared files contain paragraphs as lines; some HF modes consume per-line iterator input, while other modes alter segmentation.
2. The benchmark's `split` choices change the candidate merge boundaries and generally change the learned model. A Python Rewrite result with no pre-tokenizer is not a fair same-model comparison to HF `WhitespaceSplit` or `ByteLevel`; use the closest common token stream and state the remaining mismatch.
3. Equal `vocab_size` is not equal work. Record actual merge count, initial alphabet size, final vocabulary size, and model digest. For exact algorithm comparisons, compare the canonical ordered merge sequence and vocabulary IDs, not just file compression or runtime.
4. The local `BPETrainer` may have a different default `min_freq` (the repository's `ebpe.py` defaults to 10, while the HF harness uses 2) and its corpus assembly/normalization may differ. Explicitly set the parameter and preprocessing contract for the intended experiment. If the Rewrite baseline cannot support a particular pretokenization mode, omit that row rather than equating it to a different segmentation.
5. To distinguish implementation speed from behavior, pair the Rewrite runner against a simple reference trainer on exactly the same token stream and check merge-sequence equality; then use HF rows as an ecosystem/context baseline rather than claiming direct algorithmic equivalence where preprocessing differs.

## Suggested minimal first matrix

Reuse the existing `en-{1,4,16,32}m.txt` and `zh-{1,4,16,32}m.txt` files first; these match the benchmark's small/large scale coverage, and the README already warns that 32 MiB Chinese needs a larger target vocabulary because the initial alphabet alone can exceed 8,000. Fix one agreed tokenization policy and a pair of explicit vocabulary targets (8,000 at 1/4 MiB; 16,000 at 16/32 MiB), run each case in a separate process with three repetitions, and report median wall time, peak RSS, actual merges, actual vocab, and a canonical model digest. Add `de` and `ja` when the base runner is stable; all their samples are already present.

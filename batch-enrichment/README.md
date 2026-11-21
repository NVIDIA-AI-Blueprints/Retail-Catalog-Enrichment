# Batch catalog enrichment

## What this is

Take a catalog of product listings — messy titles, half-written descriptions, vendor SKU
noise — and turn every one of them into structured attributes: category, brand, color,
material, sizes, and sixteen more. Run it over the whole catalog as a batch job, as fast as
one GPU can go.

This directory is the batch path. For the interactive, image-driven enrichment path, see
[`../src`](../src).

You get three things:

1. **A working pipeline.** Parquet in, enriched Parquet out. ~400 lines of plain Python
   around an inference engine. Point it at your own catalog and schema.
2. **A benchmark that measures it honestly.** Throughput, validity rate, where the host CPU
   goes, and how close the whole thing gets to the engine's ceiling.
3. **Two sizing tools you can run before booking a GPU.** One tells you whether your schema
   will work and what it will cost in tokens. The other tells you how many GPUs your
   catalog needs and what that costs in dollars.

The output is guaranteed to match your JSON Schema, because the schema constrains decoding
rather than being checked afterward. That turns out to make it faster, not slower — see
Results.

```
src/catalog_bench/     the pipeline, the benchmark, the sizing tools
  engines/             trtllm | http | mock, behind one submit/drain interface
  schemas/             the reference schema (21 attributes) — swap in your own
  templates/           the prompt: invariant prefix + per-product suffix
scripts/               run it on a GPU, or serve it over HTTP and run it again
tests/                 52 checks, all CPU
```

---

## Quickstart

### On a laptop, in two minutes

No GPU, no model, no weights. This proves the whole thing works before you pay for
hardware.

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
pytest                                                  # 52 checks

# Will my schema work, and what will it cost me per product?
catalog-bench check-schema src/catalog_bench/schemas/product_attributes.schema.json \
  --tokenizer Qwen/Qwen2.5-1.5B-Instruct

# How many GPUs does my catalog need, and what does that cost?
catalog-bench capacity --catalog-size 40_000_000 --sla-hours 6 --gpu h200
```

`check-schema` is the cheapest command here and prevents the most expensive mistake: a
schema that works fine but quietly costs you three times the throughput, discovered at the
end of a ten-hour run. It exits non-zero on errors, so it drops into CI.

To rehearse the full benchmark with a simulated engine — same stages, same order, same
files, no GPU:

```bash
ENGINE=mock MODEL=mock TOKENIZER=Qwen/Qwen2.5-1.5B-Instruct \
N_PRODUCTS=2000 N_PROBE=400 IN_FLIGHT="32 128" WARMUP=50 \
  bash scripts/run_benchmark.sh
```

Those numbers are simulated and mean nothing. Every record it writes is stamped
`mock: true`, and the report refuses to draw a conclusion from one.

### On a GPU

You need one H200 141GB (or an H100 80GB — see the note below), ~110 GB of free disk, and
Docker with the NVIDIA runtime. No Hugging Face token; the model is public.

```bash
bash scripts/gpu_run.sh smoke     # one model load — does it work at all?
bash scripts/gpu_run.sh full      # ~30 min — the real thing
```

Always run `smoke` first. It costs one model load and catches the two most likely failures:
the container can't see the GPU, and your schema doesn't compile to a grammar. If you're on
a card you haven't used before, `bash scripts/gpu_run.sh fit` checks which concurrency
levels actually fit in memory — three minutes there beats an out-of-memory error ten
minutes into a full run.

**On an H100 80GB, set `IN_FLIGHT="256 512"`.** Batch 1024 doesn't fit — see
[Memory at batch size](#memory-at-batch-size).

Every knob is an environment variable — `MODEL`, `IN_FLIGHT`, `GRAMMARS`, `SCHEMA_MODE`,
`WS`, `IMAGE`. Defaults describe what the blueprint intends; your hardware's constraints
stay on the command line.

### To run it on your own catalog

Point `--catalog` at Parquet with these columns:

| column | type | |
|---|---|---|
| `product_id` | string, required | the key for retries and replay |
| `title` | string, required | |
| `description` | string, nullable | |
| `metadata` | string, nullable | free-form; passed through as-is |

and `--schema` at your own JSON Schema. Run `check-schema` on it first.

---

## Results

1× H200 141GB, `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`, TensorRT-LLM 1.3.0rc9,
4,000 products, `schema_mode=compact`, mean input 661.6 tokens.

`valid products/s` is `products/s x goodput` — the rate of output that passes schema
validation, which is the rate that matters.

| engine | constrained | in-flight | products/s | goodput | valid products/s | tok/s | host CPU |
|---|---|---|---|---|---|---|---|
| TensorRT-LLM in-process | off | 256 | 13.63 | 82.0% | 11.18 | 13,212 | 4.3% |
| TensorRT-LLM in-process | off | 512 | 23.34 | 82.2% | 19.19 | 22,646 | 6.2% |
| TensorRT-LLM in-process | off | 1024 | 42.76 | 82.4% | 35.22 | 41,460 | 8.9% |
| TensorRT-LLM in-process | xgrammar | 256 | 13.26 | 100% | 13.26 | 12,775 | 4.3% |
| TensorRT-LLM in-process | xgrammar | 512 | 22.54 | 100% | 22.54 | 21,721 | 5.9% |
| TensorRT-LLM in-process | xgrammar | 1024 | 38.31 | 100% | **38.31** | 36,896 | 7.7% |
| vLLM served | off | 256 | 17.11 | 82.5% | 14.12 | 16,588 | 8.6% |
| vLLM served | off | 512 | 25.10 | 82.3% | 20.66 | 24,399 | 13.9% |
| vLLM served | off | 1024 | 38.10 | 81.4% | 30.99 | 37,122 | 21.7% |
| vLLM served | xgrammar | 256 | 16.34 | 100% | **16.34** | 15,743 | 9.3% |
| vLLM served | xgrammar | 512 | 21.81 | 100% | 21.81 | 21,096 | 14.2% |
| vLLM served | xgrammar | 1024 | 31.65 | 100% | 31.65 | 30,659 | 21.4% |

At 38.31 valid products/s, one GPU does **137,916 products/hour**: a 100K catalog in ~44
minutes, 1M in ~7.4 hours. A 10M catalog on a 4-hour deadline needs 19 GPUs.

### Against the bare engine

`trtllm-bench` on the same token shape with no pipeline around it, unconstrained, compared
against the in-process unconstrained rows above.

| in-flight | bare engine tok/s | pipeline tok/s | pipeline % of engine |
|---|---|---|---|
| 256 | 30,201 | 13,212 | 43.7% |
| 512 | 37,394 | 22,646 | 60.6% |
| 1024 | 42,188 | 41,460 | **98.3%** |

The bare-engine dataset pins output length at 341 tokens; the pipeline's measured mean is
308.3. The gap at 256 and 512 was not investigated.

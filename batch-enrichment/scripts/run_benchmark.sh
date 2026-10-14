#!/usr/bin/env bash
# The benchmark sequence: schema check, catalog, token accounting, bare-engine ceiling,
# pipeline sweep, accuracy, report.
#
# Run from the project root, inside the TensorRT-LLM container:
#
#     MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 bash scripts/run_benchmark.sh
#
# Rehearse the whole thing on a laptop first — same stages, same order, same file
# plumbing, simulated decode. It produces NO valid measurements, but it proves the script
# works before a GPU is on the clock:
#
#     ENGINE=mock MODEL=mock TOKENIZER=Qwen/Qwen2.5-1.5B-Instruct \
#     N_PRODUCTS=2000 N_PROBE=400 IN_FLIGHT="32 128" WARMUP=50 \
#       bash scripts/run_benchmark.sh
#
# Everything is overridable by environment variable, so the defaults document what the
# blueprint intends and your host's constraints stay on the command line.
set -euo pipefail

ENGINE="${ENGINE:-trtllm}"
MODEL="${MODEL:?set MODEL to a TRT-LLM engine dir or HF id}"
TOKENIZER="${TOKENIZER:-$MODEL}"

N_PRODUCTS="${N_PRODUCTS:-10000}"
# Sized so the steady-state window stays wide even at 1024 in flight. Products are nearly
# free next to a model load, so undersizing this buys nothing.
N_PROBE="${N_PROBE:-4000}"

# `compact` is where a sane integrator lands. `full` costs ~2.8x the fleet for the same
# work; `none` is measured for contrast in the token stage without needing its own sweep.
SCHEMA_MODE="${SCHEMA_MODE:-compact}"

# An octave of concurrency, which is what shows whether throughput is still climbing.
# On an 80 GiB card use "256 512" — batch 1024 needs ~45 GiB of Mamba state on top of
# ~33 GiB of weights and will OOM. That is a property of the model, not a misconfiguration.
IN_FLIGHT="${IN_FLIGHT:-256 1024}"

# grammar=off is the like-for-like denominator for the guided rows. The bare-engine
# baseline is always unguided, so it cannot isolate grammar cost on its own.
GRAMMARS="${GRAMMARS:-off xgrammar}"

# Measured worst-case output length is 485 tokens; 512 leaves headroom without letting a
# runaway generation cost much. The schema check enforces the relationship in stage 0.
MAX_TOKENS="${MAX_TOKENS:-512}"

# Executor sizing. These describe the card, not the workload. max_seq_len must cover the
# worst case: catalog p95 product text + the invariant prefix + MAX_TOKENS.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
KV_FRACTION="${KV_FRACTION:-0.7}"

# Excluded from the rate: grammar compilation, CUDA graph capture and kernel selection,
# none of which recur. trtllm-bench warms up too, and efficiency divides one by the other.
WARMUP="${WARMUP:-200}"

SCHEMA="${SCHEMA:-src/catalog_bench/schemas/product_attributes.schema.json}"
BENCH="${BENCH:-catalog-bench}"

# Fail before a 30B model loads, not after.
if [ "$ENGINE" = "trtllm" ]; then
  python -c "import tensorrt_llm; print('tensorrt_llm', tensorrt_llm.__version__)" || {
    echo "tensorrt_llm not importable — are you inside the TRT-LLM container?"; exit 1; }
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "no GPU visible"; exit 1; }
fi

echo "=== 0/6  schema check (before a 30B model loads)"
$BENCH check-schema "$SCHEMA" --tokenizer "$TOKENIZER" --max-tokens "$MAX_TOKENS"

echo "=== 1/6  synthetic catalog (${N_PRODUCTS} products) + ground truth"
$BENCH catalog --n "$N_PRODUCTS" --seed 7 --out data/catalog \
  --ground-truth data/ground_truth.parquet

echo "=== 2/6  token accounting + bare-engine dataset"
# All three schema modes: the spread between them is a ~3.7x spread in tokens per
# product, which is a ~3.7x spread in throughput.
for mode in none compact full; do
  $BENCH tokens --catalog data/catalog --tokenizer "$TOKENIZER" --schema-mode "$mode" \
    --out "reports/tokens_${mode}.json" --bench-dataset "data/bench_${mode}.jsonl"
done

echo "=== 3/6  bare-engine ceiling"
if [ "$ENGINE" != "trtllm" ]; then
  echo "  skipped: needs a GPU (there is nothing to simulate)"
else
  # The same dataset the pipeline will run, so the ratio compares like with like.
  for c in $IN_FLIGHT; do
    $BENCH sol --model "$MODEL" --dataset "data/bench_${SCHEMA_MODE}.jsonl" \
      --concurrency "$c" --num-requests "$N_PROBE" \
      --bench-arg "max_batch_size=$c" \
      --bench-arg "max_seq_len=$MAX_SEQ_LEN" \
      --bench-arg "kv_cache_free_gpu_mem_fraction=$KV_FRACTION" \
      --out "reports/sol_baseline_c${c}.json" \
      --raw-report "reports/trtllm_bench_raw_c${c}.json"
  done
fi

echo "=== 4/6  pipeline sweep"
for g in $GRAMMARS; do
  for f in $IN_FLIGHT; do
    echo "--- grammar=$g in-flight=$f"
    # max_batch_size tracks the in-flight level because on this Mamba hybrid the
    # per-sequence state, not the KV cache, is what scales with batch. trtllm only —
    # the other engines reject these.
    TRT_ARGS=""
    if [ "$ENGINE" = "trtllm" ]; then
      TRT_ARGS="--trtllm-kwarg max_batch_size=$f \
                --trtllm-kwarg max_seq_len=$MAX_SEQ_LEN \
                --trtllm-kwarg kv_cache_config={\"free_gpu_memory_fraction\":$KV_FRACTION}"
    fi
    # shellcheck disable=SC2086
    $BENCH run --catalog data/catalog --engine "$ENGINE" --model "$MODEL" \
      --tokenizer "$TOKENIZER" --grammar "$g" --schema-mode "$SCHEMA_MODE" \
      --n "$N_PROBE" --in-flight "$f" --warmup "$WARMUP" \
      --max-tokens "$MAX_TOKENS" $TRT_ARGS \
      --save-outputs "data/enriched_${g}_if${f}.parquet"
  done
done

echo "=== 5/6  accuracy (synthetic labels: regression signal, not an accuracy claim)"
# Scored on the guided run at the highest concurrency — the configuration you would ship.
# Grammar-off output is deliberately not scored: its failures are parse errors, which
# goodput already reports.
EVAL_GRAMMAR="${EVAL_GRAMMAR:-xgrammar}"
EVAL_IN_FLIGHT="${IN_FLIGHT##* }"
EVAL_FILE="data/enriched_${EVAL_GRAMMAR}_if${EVAL_IN_FLIGHT}.parquet"
if [ -f "$EVAL_FILE" ]; then
  $BENCH accuracy --outputs "$EVAL_FILE" --ground-truth data/ground_truth.parquet
else
  echo "  skipped: $EVAL_FILE not found (was $EVAL_GRAMMAR in \$GRAMMARS?)"
fi

echo "=== 6/6  report + capacity"
$BENCH report reports --out reports/benchmark_report.md
$BENCH capacity --from-reports reports --catalog-size 10_000_000 --sla-hours 4

echo
echo "Report: reports/benchmark_report.md"
echo
echo "If the node is still yours, the highest-value follow-ups:"
echo "  1. KV prefix reuse on vs off. ~70% of every request is a byte-identical prefix,"
echo "     but TRT-LLM downgrades block reuse when SSM layers are present, so it may"
echo "     never be reused at all. If toggling it moves nothing, prefix caching is not"
echo "     an available optimization for this model family."
echo "  2. llguidance vs xgrammar:   GRAMMARS=\"xgrammar llguidance\""
echo "  3. schema_mode vs accuracy:  SCHEMA_MODE=none, then compare accuracy"
echo "  4. concurrency above 1024, and multi-GPU scale-out (never measured here)"

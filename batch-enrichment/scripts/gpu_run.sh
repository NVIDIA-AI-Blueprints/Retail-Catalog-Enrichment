#!/usr/bin/env bash
# Run the benchmark inside the TensorRT-LLM container.
#
#     bash scripts/gpu_run.sh smoke     # ~1 model load. Does the plumbing work at all?
#     bash scripts/gpu_run.sh fit       # ~3 min. Does the executor fit at each batch size?
#     bash scripts/gpu_run.sh full      # ~30 min + six model loads. The whole sequence.
#
# Run `smoke` before anything else. It costs one model load and catches the two most
# likely failures: the container cannot see the GPU, and the schema does not compile to a
# grammar. Run `fit` before `full` if you are on a card you have not used before —
# finding out that batch 1024 does not fit costs three minutes here against ten minutes
# into a six-stage run.
#
# WS is the directory containing this checkout. Override it for your host.
set -uo pipefail

MODE="${1:-smoke}"
WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT="${PROJECT:-$(basename "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)")}"
IMAGE="${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9}"
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8}"
HF_HOME="${HF_HOME:-$WS/.cache/huggingface}"

# Batch 1024 needs ~45 GiB of Mamba state on top of ~33 GiB of weights, so it fits on a
# 141 GiB H200 and does not fit on an 80 GiB H100. Set IN_FLIGHT="256 512" there.
IN_FLIGHT="${IN_FLIGHT:-256 1024}"

# Executor sizing. Without these, LLM() sizes itself from model defaults and asks for
# 92 GiB regardless of the card.
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
KV_FRACTION="${KV_FRACTION:-0.7}"

DEPS="pyarrow jinja2 jsonschema tokenizers psutil numpy requests"

case "$MODE" in
  smoke)
    # in-flight 8, 50 products, both grammar settings. Tests that it runs and that the
    # schema compiles; the numbers mean nothing at this size. Discarded via --out
    # /dev/null for the same reason as `fit`: `full` reports over everything in reports/,
    # and the documented workflow runs smoke first, so a kept record would fold two
    # 50-product warmup-dominated rows into a matrix of 4000-product measurements.
    CMD="
      catalog-bench catalog --n 200 --seed 7 --out data/catalog
      for G in off xgrammar; do
        echo \"--- smoke grammar=\$G\"
        catalog-bench run --engine trtllm --model $MODEL --grammar \$G \\
          --n 50 --in-flight 8 --warmup 10 --out /dev/null \\
          --trtllm-kwarg max_batch_size=8 \\
          --trtllm-kwarg max_seq_len=$MAX_SEQ_LEN \\
          --trtllm-kwarg kv_cache_config='{\"free_gpu_memory_fraction\":$KV_FRACTION}'
      done"
    ;;
  fit)
    # Allocation only, not throughput. Small --n on purpose: the steady-state window is
    # meaningless here and these numbers must not be reported. Descending order so the
    # most likely failure is hit first.
    LEVELS=$(echo "$IN_FLIGHT" | tr ' ' '\n' | sort -rn | tr '\n' ' ')
    CMD="
      catalog-bench catalog --n 200 --seed 7 --out data/catalog
      for F in $LEVELS; do
        echo \"--- fit check at in-flight=\$F\"
        catalog-bench run --engine trtllm --model $MODEL --grammar xgrammar \\
          --n 64 --in-flight \$F --warmup 0 --out /dev/null \\
          --trtllm-kwarg max_batch_size=\$F \\
          --trtllm-kwarg max_seq_len=$MAX_SEQ_LEN \\
          --trtllm-kwarg kv_cache_config='{\"free_gpu_memory_fraction\":$KV_FRACTION}' \\
          && echo \"    in-flight \$F FITS\" || echo \"    in-flight \$F DOES NOT FIT\"
      done"
    ;;
  full)
    CMD="MODEL=$MODEL IN_FLIGHT='$IN_FLIGHT' MAX_SEQ_LEN=$MAX_SEQ_LEN \
         KV_FRACTION=$KV_FRACTION bash scripts/run_benchmark.sh"
    ;;
  *)
    echo "usage: $0 {smoke|fit|full}" >&2
    exit 2
    ;;
esac

echo "mode=$MODE  image=$IMAGE  model=$MODEL  workspace=$WS/$PROJECT"

docker run --rm --gpus all \
  -v "$WS":/workspace \
  -v "$HF_HOME":/root/.cache/huggingface \
  -w "/workspace/$PROJECT" \
  "$IMAGE" bash -lc "
    set -e
    ulimit -n 65536
    pip install -q $DEPS 2>&1 | tail -1
    pip install -q -e . 2>&1 | tail -1
    $CMD
  "

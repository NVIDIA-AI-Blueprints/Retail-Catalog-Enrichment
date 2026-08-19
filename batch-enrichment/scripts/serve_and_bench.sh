#!/usr/bin/env bash
# Serve the model over HTTP and run the same sweep against it.
#
# The served arm of the comparison: identical prompts, identical schema, identical sweep,
# reaching the GPU through an OpenAI-compatible server instead of in-process. Records land
# in reports/ alongside the in-process ones and the report keeps them in separate rows.
#
#     bash scripts/serve_and_bench.sh
#     MODEL=... PORT=8002 TAG=lightning bash scripts/serve_and_bench.sh
#
# TAG suffixes every output path, so two models can be benchmarked without colliding.
#
# ONE THING TO CHECK BEFORE BELIEVING ANY NUMBER FROM THIS. The client's thread pool caps
# real concurrency. If it is below --in-flight, the whole sweep runs at the pool size,
# produces flat throughput, and reads like a finding about the server's scheduler. Our own
# `in_flight_mean` does NOT catch this — it counts submitted futures, most of them queued
# client-side. The server's own counter is the check:
#
#     grep -o "Running: [0-9]* reqs" reports/server${TAG:+_$TAG}.log | sort -u | tail -1
#
# Peak `Running:` should approach --in-flight. `Running: 128, Waiting: 0` when you asked
# for 1024 is a starved server and the numbers are meaningless.
set -uo pipefail

WS="${WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT="${PROJECT:-$(basename "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)")}"
HF_HOME="${HF_HOME:-$WS/.cache/huggingface}"

MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8}"
SERVE_IMAGE="${SERVE_IMAGE:-vllm/vllm-openai:latest}"
CLIENT_IMAGE="${CLIENT_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9}"
PORT="${PORT:-8001}"
TAG="${TAG:-}"
SUFFIX="${TAG:+_$TAG}"
CONTAINER="catalog_bench_server${SUFFIX}"

IN_FLIGHT="${IN_FLIGHT:-256 512 1024}"
GRAMMARS="${GRAMMARS:-off xgrammar}"
N_PROBE="${N_PROBE:-4000}"
WARMUP="${WARMUP:-200}"
SCHEMA_MODE="${SCHEMA_MODE:-compact}"
MAX_TOKENS="${MAX_TOKENS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"

URL="http://localhost:$PORT"
LOG="$WS/$PROJECT/reports/server${SUFFIX}.log"

log() { printf '\n\033[1;34m[%s] ==> %s\033[0m\n' "$(date '+%T')" "$1"; }
cleanup() {
  docker logs "$CONTAINER" >> "$LOG" 2>&1 || true
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$(dirname "$LOG")"

log "Starting server  model=$MODEL  port=$PORT"
# No --rm: we want the logs if it crashes. Pin the grammar backend explicitly — there is
# no runtime API to ask which one it chose, so the startup log is the only record.
docker run -d --name "$CONTAINER" \
  --gpus all --ipc=host \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -p "$PORT:8000" \
  -v "$HF_HOME":/root/.cache/huggingface \
  "$SERVE_IMAGE" \
  "$MODEL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --structured-outputs-config '{"backend":"xgrammar"}' >/dev/null || {
    echo "!! failed to start the server container"; exit 1; }

log "Waiting for the server (up to ${READY_TIMEOUT}s — a 30B model takes a while)"
for i in $(seq 1 "$READY_TIMEOUT"); do
  if curl -sf "$URL/health" >/dev/null 2>&1; then
    echo "ready after ${i}s"
    break
  fi
  STATUS=$(docker inspect --format '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo gone)
  case "$STATUS" in
    exited|dead|gone)
      echo "!! server exited early (after ${i}s). Log tail:"
      docker logs --tail 40 "$CONTAINER" 2>&1 || true
      exit 1
      ;;
  esac
  [ $((i % 30)) -eq 0 ] && echo "  still waiting... ${i}s"
  sleep 1
  [ "$i" -eq "$READY_TIMEOUT" ] && {
    echo "!! server did not become ready in ${READY_TIMEOUT}s"
    docker logs --tail 40 "$CONTAINER" 2>&1 || true
    exit 1
  }
done

docker logs "$CONTAINER" > "$LOG" 2>&1 || true
echo "startup log: $LOG"
grep -iE "backend|xgrammar|guided|error|warning" "$LOG" | head -20 || true

log "Running the sweep against $URL"
docker run --rm --network host \
  -v "$WS":/workspace \
  -v "$HF_HOME":/root/.cache/huggingface \
  -w "/workspace/$PROJECT" \
  "$CLIENT_IMAGE" bash -lc "
    set -e
    ulimit -n 65536
    pip install -q pyarrow jinja2 jsonschema tokenizers psutil numpy requests 2>&1 | tail -1
    pip install -q -e . 2>&1 | tail -1
    for G in $GRAMMARS; do
      for F in $IN_FLIGHT; do
        echo \"=== served grammar=\$G in-flight=\$F \$(date '+%T') ===\"
        catalog-bench run --engine http --http-base-url $URL \\
          --model '$MODEL' --tokenizer '$MODEL' \\
          --grammar \$G --schema-mode $SCHEMA_MODE \\
          --n $N_PROBE --in-flight \$F --warmup $WARMUP --max-tokens $MAX_TOKENS \\
          --save-outputs 'data/enriched_served${SUFFIX}_'\$G'_if'\$F'.parquet' \\
          --out 'reports/probe_served${SUFFIX}_'\$G'_if'\$F'.json'
      done
    done
    catalog-bench report reports --out 'reports/benchmark_report${SUFFIX}.md'
  "
RC=$?

log "Peak server-side concurrency (the number that matters)"
grep -o "Running: [0-9]* reqs" "$LOG" | sort -u | tail -1 || \
  echo "  (no Running: counter in the log — verify concurrency another way)"

echo "### finished rc=$RC"
exit $RC

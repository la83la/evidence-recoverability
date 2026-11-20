#!/usr/bin/env bash
# Reproduce the long-term-memory recoverability pilot (LongMemEval_S + LoCoMo).
# Usage: bash memory_poc/run_all.sh [OUT_DIR] [N_BASE_QUESTIONS]
#   bash memory_poc/run_all.sh out_mem            # full run (~15 min on one RTX 3080)
#   bash memory_poc/run_all.sh out_mem_smoke 40   # smoke test
# Env: PY=python DEVICE=cuda:0 BATCH=8 MODEL=Qwen/Qwen2.5-3B-Instruct REVISION=<hf commit> K1=10 K2=10
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
OUT="${1:-$ROOT/out_mem}"; N="${2:-}"
PY="${PY:-python}"; DEVICE="${DEVICE:-cuda:0}"; BATCH="${BATCH:-8}"; K1="${K1:-10}"; K2="${K2:-10}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
REVISION="${REVISION:-aa8e72537993ba99e69dfaafa59ed015b17504d1}"   # only valid for the default model
DATA="$ROOT/memory_data"; mkdir -p "$DATA" "$OUT"
[ -f "$DATA/longmemeval_s_cleaned.json" ] || curl -L --fail -o "$DATA/longmemeval_s_cleaned.json" \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
[ -f "$DATA/locomo10.json" ] || curl -L --fail -o "$DATA/locomo10.json" \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
( cd "$DATA" && sha256sum -c - <<SUMS
d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442  longmemeval_s_cleaned.json
79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4  locomo10.json
SUMS
)
REVARG=(); [ "$MODEL" = "Qwen/Qwen2.5-3B-Instruct" ] && REVARG=(--revision "$REVISION")
cd "$HERE"
for ds in longmemeval_s locomo; do
  f=longmemeval_s_cleaned.json; [ "$ds" = locomo ] && f=locomo10.json
  mkdir -p "$OUT/$ds"; mkdir -p "$OUT/$ds"
  "$PY" run_memory.py --dataset "$ds" --data "$DATA/$f" --out "$OUT/$ds" --cf_delete \
     --model "$MODEL" "${REVARG[@]}" --k1 "$K1" --k2 "$K2" --batch "$BATCH" --device "$DEVICE" ${N:+--n "$N"} \
     2>&1 | grep -vE "it/s\]" | tee "$OUT/$ds/run.log"
done
"$PY" features_memory.py "$OUT/longmemeval_s" "$OUT/locomo" | tee "$OUT/features.log"
"$PY" sample_audit_memory.py "$OUT/longmemeval_s" > "$OUT/longmemeval_s/audit_sample.txt"
"$PY" sample_audit_memory.py "$OUT/locomo" > "$OUT/locomo/audit_sample.txt"
echo "done -> $OUT   (compare with $HERE/reference_results/)"

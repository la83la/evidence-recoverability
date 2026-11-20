#!/usr/bin/env bash
# One-command reproduction of the MuSiQue recoverability pilot.
# Usage: bash run_all.sh [OUT_DIR] [N_QUESTIONS]      e.g. bash run_all.sh out_full        (all 2417)
#                                                          bash run_all.sh out_smoke 50     (smoke test)
# Env:   DEVICE=cuda:0|auto  BATCH=16  PY=python  MODEL=Qwen/Qwen2.5-3B-Instruct
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/out_full}"; N="${2:-}"
PY="${PY:-python}"; DEVICE="${DEVICE:-cuda:0}"; BATCH="${BATCH:-16}"; MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DATA="$HERE/data/musique_ans_v1.0_dev.jsonl"
SHA=15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b
mkdir -p "$HERE/data" "$OUT"
if [ ! -f "$DATA" ]; then
  curl -L --fail -o "$DATA" https://huggingface.co/datasets/bdsaglam/musique/resolve/main/musique_ans_v1.0_dev.jsonl
fi
echo "$SHA  $DATA" | sha256sum -c -
"$PY" "$HERE/run_pilot.py" --data "$DATA" --out "$OUT" --batch "$BATCH" --device "$DEVICE" --model "$MODEL" ${N:+--n "$N"} 2>&1 | grep -vE "^gen |it/s\]" | tee "$OUT/run.log"
"$PY" "$HERE/features.py" "$OUT" | tee "$OUT/features.log"
"$PY" "$HERE/sample_audit.py" "$OUT" > "$OUT/audit_sample.txt"
echo "done -> $OUT (compare with $HERE/reference_results/)"

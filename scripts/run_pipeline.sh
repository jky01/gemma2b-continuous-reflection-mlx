#!/usr/bin/env bash
# 端對端執行腳本：Phase 1 → Phase 3 → Phase 4
# 使用方式：
#   bash scripts/run_pipeline.sh
# 或指定模型路徑：
#   MODEL_PATH=google/gemma-2-2b bash scripts/run_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-google/gemma-2-2b}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"

ADAPTER_OUT="results/adapter_weights.safetensors"

echo "════════════════════════════════════════"
echo "  Gemma 2B 連續反思迴圈訓練 Pipeline"
echo "  Model : ${MODEL_PATH}"
echo "  Epochs: ${EPOCHS}  BS: ${BATCH_SIZE}  LR: ${LR}"
echo "════════════════════════════════════════"

# ── Phase 1: Baseline ─────────────────────────────────────────────────────────
echo ""
echo "▶ Phase 1: Baseline 測試（sample_qa.jsonl）"
python -m src.baseline \
    --model-path   "${MODEL_PATH}" \
    --dataset      data/sample_qa.jsonl \
    --max-length   "${MAX_LENGTH}"

# ── Phase 3: Training ─────────────────────────────────────────────────────────
echo ""
echo "▶ Phase 3: Adapter 訓練"
python -m src.train \
    --model-path    "${MODEL_PATH}" \
    --train-dataset data/train.jsonl \
    --val-dataset   data/test.jsonl \
    --epochs        "${EPOCHS}" \
    --batch-size    "${BATCH_SIZE}" \
    --lr            "${LR}" \
    --max-length    "${MAX_LENGTH}" \
    --output        "${ADAPTER_OUT}"

# ── Phase 4: Evaluation ───────────────────────────────────────────────────────
echo ""
echo "▶ Phase 4: 評估與分析"
python -m src.evaluate \
    --model-path      "${MODEL_PATH}" \
    --adapter-weights "${ADAPTER_OUT}" \
    --dataset         data/test.jsonl \
    --temperature     "${TEMPERATURE}" \
    --top-p           "${TOP_P}" \
    --max-length      "${MAX_LENGTH}"

echo ""
echo "════════════════════════════════════════"
echo "  Pipeline 完成！結果存放於 results/"
echo "════════════════════════════════════════"

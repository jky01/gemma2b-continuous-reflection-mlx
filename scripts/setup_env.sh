#!/usr/bin/env bash
# 建立虛擬環境並安裝 MLX / MLX-LM 相關依賴（適用於 Apple Silicon, 如 M2 MacBook Air）
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "==> 建立虛擬環境 (.venv)"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

echo "==> 升級 pip"
pip install --upgrade pip

echo "==> 安裝套件 (requirements.txt)"
pip install -r requirements.txt

echo "==> 下載 Gemma 2B 權重（首次執行需要登入 Hugging Face CLI: huggingface-cli login）"
echo "    例如: mlx_lm.convert --hf-path google/gemma-2-2b -q"

echo "==> 環境建置完成。請執行: source .venv/bin/activate"

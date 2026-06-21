"""
Phase 1: 環境建置與基底準備 (Environment & Baseline)

目標：
1. 確認 MLX / mlx-lm 生態系運作正常（M2 GPU 加速）。
2. 載入 Gemma 2B 權重，凍結 Transformer 主體參數。
3. 在小型推理資料集上跑單向 baseline，記錄平均 Cross-Entropy Loss 與記憶體峰值佔用量。

執行方式：
    python src/baseline.py --model-path google/gemma-2-2b --dataset data/sample_qa.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_model(model_path: str):
    """載入 Gemma 2B 並凍結主體權重。

    MLX 預設不追蹤梯度；凍結的重點在於 Phase 3 value_and_grad 時，
    只把 Adapter 的參數傳入（base_model 已 freeze() 標記為不可訓練）。
    """
    from mlx_lm import load
    model, tokenizer = load(model_path)
    model.freeze()
    return model, tokenizer


def load_dataset(dataset_path: str):
    """載入小型邏輯推理資料集（例如 GSM8K 子集或自建 QA）。

    預期格式：每行一個 JSON，至少包含 {"prompt": ..., "target": ...}
    """
    examples = []
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run_baseline(model, tokenizer, examples):
    """跑單向（single forward pass）推論，記錄平均 Cross-Entropy Loss 與記憶體峰值。

    採用 next-token prediction loss：logits[t] 預測 input_ids[t+1]。
    """
    total_loss = 0.0
    peak_memory_bytes = 0

    for ex in examples:
        text = f"{ex['prompt']} {ex.get('target', '')}"
        ids = mx.array(tokenizer.encode(text))[None]  # (1, seq_len)

        logits, _ = model(ids)  # (1, seq_len, vocab_size)

        # Next-token prediction：position t 預測 position t+1
        aligned_logits = logits[:, :-1, :]   # (1, seq_len-1, vocab_size)
        aligned_targets = ids[:, 1:]          # (1, seq_len-1)

        loss = nn.losses.cross_entropy(aligned_logits, aligned_targets, reduction="mean")
        mx.eval(loss)
        total_loss += loss.item()

        if hasattr(mx, "metal"):
            mem = mx.metal.get_active_memory()
            if mem > peak_memory_bytes:
                peak_memory_bytes = mem

    avg_loss = total_loss / max(len(examples), 1)
    return {
        "avg_cross_entropy_loss": round(avg_loss, 6),
        "peak_memory_mb": round(peak_memory_bytes / (1024 * 1024), 2),
        "num_examples": len(examples),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline 測試")
    parser.add_argument("--model-path", default="google/gemma-2-2b")
    parser.add_argument("--dataset", default="data/sample_qa.jsonl")
    args = parser.parse_args()

    start = time.time()
    model, tokenizer = load_model(args.model_path)
    examples = load_dataset(args.dataset)
    metrics = run_baseline(model, tokenizer, examples)
    elapsed = time.time() - start

    print("=== Phase 1 Baseline 結果 ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"耗時: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

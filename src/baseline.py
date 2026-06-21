"""
Phase 1: 環境建置與基底準備 (Environment & Baseline)

目標：
1. 確認 MLX / mlx-lm 生態系運作正常（M2 GPU 加速）。
2. 載入 Gemma 2B 權重，凍結 Transformer 主體參數。
3. 在小型推理資料集上跑單向 baseline，記錄平均 Cross-Entropy Loss、
   Perplexity 與記憶體峰值佔用量。

執行方式：
    python -m src.baseline --model-path google/gemma-2-2b --dataset data/sample_qa.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from .data_utils import compute_perplexity, load_jsonl, next_token_loss, tokenize_example


def load_model(model_path: str):
    """載入 Gemma 2B 並凍結主體權重。"""
    from mlx_lm import load
    model, tokenizer = load(model_path)
    model.freeze()
    return model, tokenizer


def run_baseline(model, tokenizer, examples, max_length: int = 256):
    """跑單向（single forward pass）推論，記錄平均 Cross-Entropy Loss、Perplexity 與記憶體峰值。"""
    total_loss = 0.0
    peak_memory_bytes = 0
    skipped = 0

    for ex in examples:
        ids = tokenize_example(tokenizer, ex["prompt"], ex.get("target", ""), max_length)
        if len(ids) < 2:
            skipped += 1
            continue

        input_ids = mx.array(ids)[None]  # (1, seq_len)
        loss = next_token_loss(model, input_ids)
        mx.eval(loss)
        total_loss += loss.item()

        if hasattr(mx, "metal"):
            mem = mx.metal.get_active_memory()
            if mem > peak_memory_bytes:
                peak_memory_bytes = mem

    n = max(len(examples) - skipped, 1)
    avg_loss = total_loss / n
    return {
        "num_examples":           len(examples),
        "num_skipped":            skipped,
        "avg_cross_entropy_loss": round(avg_loss, 6),
        "perplexity":             round(compute_perplexity(avg_loss), 4),
        "peak_memory_mb":         round(peak_memory_bytes / (1024 * 1024), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline 測試")
    parser.add_argument("--model-path",  default="google/gemma-2-2b")
    parser.add_argument("--dataset",     default="data/sample_qa.jsonl")
    parser.add_argument("--max-length",  type=int, default=256)
    args = parser.parse_args()

    start = time.time()
    model, tokenizer = load_model(args.model_path)
    examples = load_jsonl(args.dataset)
    metrics  = run_baseline(model, tokenizer, examples, args.max_length)
    elapsed  = time.time() - start

    print("=== Phase 1 Baseline 結果 ===")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"耗時: {elapsed:.2f}s")

    out_path = Path("results/baseline_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"elapsed_s": round(elapsed, 2), **metrics}, f, indent=2)
    print(f"結果已寫入 {out_path}")


if __name__ == "__main__":
    main()

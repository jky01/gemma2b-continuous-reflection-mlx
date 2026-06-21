"""
資料載入與前處理工具模組

提供統一的資料管線供 baseline.py / train.py / evaluate.py 使用：
    load_jsonl        — 讀取 JSONL 格式資料集
    tokenize_example  — 將單一 (prompt, target) 對轉成 token id 序列
    build_batches     — 依 batch_size 切批並 zero-pad 對齊
    compute_perplexity — 將 cross-entropy loss 換算為 perplexity
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def load_jsonl(path: str) -> List[dict]:
    """讀取每行一個 JSON 物件的資料集，回傳 list[dict]。"""
    examples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def tokenize_example(
    tokenizer,
    prompt: str,
    target: str = "",
    max_length: Optional[int] = None,
) -> List[int]:
    """將 prompt + target 串接後 tokenize，可選擇截斷。"""
    text = f"{prompt} {target}".strip()
    ids = tokenizer.encode(text)
    if max_length and len(ids) > max_length:
        ids = ids[:max_length]
    return ids


def build_batches(
    examples: List[dict],
    tokenizer,
    batch_size: int,
    max_length: int = 256,
) -> List[Tuple[mx.array, mx.array]]:
    """將 JSONL 範例轉成 (input_ids, targets) batch 列表。

    - 每個 batch 內序列 zero-pad 到相同長度。
    - 短於 2 個 token 的序列自動濾除（需至少 1 輸入 + 1 目標）。
    - 訓練時 targets == input_ids（causal language modeling）。

    Returns:
        List of (input_ids, targets) where each has shape (batch, seq_len).
    """
    pad_id = getattr(tokenizer, "pad_id", 0)

    sequences = []
    for ex in examples:
        ids = tokenize_example(
            tokenizer,
            ex["prompt"],
            ex.get("target", ""),
            max_length=max_length,
        )
        if len(ids) >= 2:
            sequences.append(ids)

    batches = []
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i : i + batch_size]
        max_len = max(len(ids) for ids in chunk)
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in chunk]
        arr = mx.array(padded)        # (batch, seq_len)
        batches.append((arr, arr))    # targets == input_ids

    return batches


def compute_perplexity(avg_cross_entropy_loss: float) -> float:
    """將平均 cross-entropy loss（以自然對數為底）轉換成 perplexity。

    Perplexity = exp(avg_loss)；值越低代表模型對資料的預測越確定。
    """
    return math.exp(avg_cross_entropy_loss)


def next_token_loss(
    model,
    input_ids: mx.array,
) -> mx.array:
    """對單一 batch 計算 next-token prediction cross-entropy loss（供 baseline 使用）。

    Args:
        model:     mlx-lm 的 Gemma 模型（已 freeze）
        input_ids: shape (batch, seq_len)

    Returns:
        scalar loss
    """
    logits, _ = model(input_ids)                # (batch, seq_len, vocab)
    aligned_logits  = logits[:, :-1, :]         # (batch, seq_len-1, vocab)
    aligned_targets = input_ids[:, 1:]          # (batch, seq_len-1)
    return nn.losses.cross_entropy(aligned_logits, aligned_targets, reduction="mean")

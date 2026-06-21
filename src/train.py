"""
Phase 3: 訓練迴圈建構 (Training Loop & Fine-tuning)

規則：
    - 只針對第二次 Pass 輸出中、對應原始序列位置的 logits 計算 Cross-Entropy Loss
    - 第一次 Pass 不計算任何 Loss
    - 使用 mlx.nn.value_and_grad，梯度只更新 Adapter（base_model 需預先 freeze）
    - 從 batch size = 1 開始測試記憶體水位（24GB M2 Air，避免觸發 swap）

執行方式：
    python -m src.train \
        --model-path google/gemma-2-2b \
        --train-dataset data/train.jsonl \
        --val-dataset   data/test.jsonl \
        --epochs 3 --batch-size 1 --lr 1e-4 --max-length 256
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .checkpoint import load_checkpoint, save_checkpoint
from .data_utils import build_batches, compute_perplexity, load_jsonl
from .model_wrapper import ReflectiveGemma, ReflectiveGemmaConfig


# ─── Loss ─────────────────────────────────────────────────────────────────────

def loss_fn(model: ReflectiveGemma, input_ids: mx.array, targets: mx.array) -> mx.array:
    """只計算第二次 pass 中、對應原始序列位置的 cross entropy loss。

    第二次 pass 的 logits 形狀為 (batch, N_v + seq_len, vocab_size)。
    前 N_v 個位置是虛擬代幣輸出，不應計入 loss。
    只取後 seq_len 個位置，並做 next-token prediction 偏移：
        logits[:, N_v + t] 預測 targets[:, t + 1]
    """
    n_v = model.config.num_virtual_tokens
    logits = model(input_ids)           # (batch, N_v + seq_len, vocab_size)

    aligned_logits  = logits[:, n_v:-1, :]   # (batch, seq_len-1, vocab_size)
    aligned_targets = targets[:, 1:]          # (batch, seq_len-1)

    return nn.losses.cross_entropy(aligned_logits, aligned_targets, reduction="mean")


# ─── Gradient utilities ───────────────────────────────────────────────────────

def clip_grad_norm(grads, max_norm: float):
    """將梯度 L2 norm 裁剪至 max_norm，防止梯度爆炸。

    使用 lazy evaluation 讓裁剪留在計算圖中，一起在 mx.eval 時批次計算。
    """
    flat_grads = [g for _, g in tree_flatten(grads)]
    total_norm = mx.sqrt(sum(mx.sum(g * g) for g in flat_grads))
    # 只在超出時縮放；scale ∈ (0, 1]
    scale = max_norm / mx.maximum(total_norm, mx.array(max_norm))
    return tree_map(lambda g: g * scale, grads)


# ─── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: ReflectiveGemma,
    optimizer,
    batches: List[Tuple[mx.array, mx.array]],
    epoch: int,
    max_grad_norm: float = 1.0,
) -> float:
    """跑一個 epoch，回傳平均訓練 loss。"""
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    running_loss = 0.0
    num_steps = 0
    for num_steps, (input_ids, targets) in enumerate(batches, start=1):
        loss, grads = loss_and_grad_fn(model, input_ids, targets)

        # 梯度裁剪
        if max_grad_norm > 0:
            grads = clip_grad_norm(grads, max_grad_norm)

        # nn.value_and_grad 只對 trainable_parameters（Adapter）求梯度
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        running_loss += loss.item()
        if (num_steps - 1) % 10 == 0:
            peak_mb = mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx, "metal") else -1
            lr = optimizer.learning_rate.item() if hasattr(optimizer.learning_rate, "item") else optimizer.learning_rate
            print(
                f"  epoch {epoch} step {num_steps - 1:4d}: "
                f"loss={loss.item():.4f}  lr={lr:.2e}  peak_mb={peak_mb:.1f}"
            )

    return running_loss / max(num_steps, 1)


def evaluate_loss(
    model: ReflectiveGemma,
    batches: List[Tuple[mx.array, mx.array]],
) -> float:
    """對驗證集計算平均 loss（不更新參數）。"""
    total_loss = 0.0
    num_steps = 0
    for num_steps, (input_ids, targets) in enumerate(batches, start=1):
        loss = loss_fn(model, input_ids, targets)
        mx.eval(loss)
        total_loss += loss.item()
    return total_loss / max(num_steps, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: 訓練 Adapter")
    parser.add_argument("--model-path",     default="google/gemma-2-2b")
    parser.add_argument("--train-dataset",  default="data/train.jsonl")
    parser.add_argument("--val-dataset",    default="data/test.jsonl")
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--batch-size",     type=int,   default=1)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--warmup-steps",   type=int,   default=10,
                        help="Linear LR warmup 步數，之後轉 cosine decay")
    parser.add_argument("--max-length",     type=int,   default=256,
                        help="單一序列最大 token 數（超過截斷）")
    parser.add_argument("--max-grad-norm",  type=float, default=1.0,
                        help="梯度 L2 norm 上限（0 = 不裁剪）")
    parser.add_argument("--output",         default="results/adapter_weights.safetensors")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints",
                        help="每個 epoch 儲存 checkpoint 的目錄")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="自動從最新 checkpoint 繼續訓練（--no-resume 從頭開始）")
    args = parser.parse_args()

    from mlx_lm import load
    base_model, tokenizer = load(args.model_path)
    base_model.freeze()

    hidden_size = base_model.model.embed_tokens.weight.shape[-1]
    config = ReflectiveGemmaConfig(hidden_size=hidden_size)
    model = ReflectiveGemma(base_model, config)

    train_examples = load_jsonl(args.train_dataset)
    val_examples   = load_jsonl(args.val_dataset) if Path(args.val_dataset).exists() else []

    train_batches = build_batches(train_examples, tokenizer, args.batch_size, args.max_length)
    val_batches   = build_batches(val_examples,   tokenizer, args.batch_size, args.max_length)

    total_steps = len(train_batches) * args.epochs

    # LR schedule：linear warmup → cosine decay
    if args.warmup_steps > 0 and total_steps > args.warmup_steps:
        lr_schedule = optim.join_schedules(
            [
                optim.linear_schedule(0.0, args.lr, args.warmup_steps),
                optim.cosine_decay(args.lr, total_steps - args.warmup_steps),
            ],
            [args.warmup_steps],
        )
    else:
        lr_schedule = args.lr

    optimizer = optim.Adam(learning_rate=lr_schedule)

    # ── Checkpoint 恢復 ────────────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    history: list = []

    if args.resume:
        start_epoch, global_step, history = load_checkpoint(
            model, optimizer, args.checkpoint_dir
        )

    if start_epoch >= args.epochs:
        print(
            f"訓練已完成（start_epoch={start_epoch} >= epochs={args.epochs}）。"
            f"若要繼續訓練，請增加 --epochs。"
        )
        return

    # ── 訓練迴圈 ───────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        train_loss = train_one_epoch(
            model, optimizer, train_batches, epoch, args.max_grad_norm
        )
        global_step += len(train_batches)

        record = {"epoch": epoch, "train_loss": train_loss, "train_ppl": compute_perplexity(train_loss)}

        if val_batches:
            val_loss = evaluate_loss(model, val_batches)
            record["val_loss"] = val_loss
            record["val_ppl"]  = compute_perplexity(val_loss)
            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_ppl={record['val_ppl']:.2f}")
        else:
            print(f"  train_loss={train_loss:.4f}  train_ppl={record['train_ppl']:.2f}")

        history.append(record)

        # 每個 epoch 存完整 checkpoint（adapter + optimizer state + metadata）
        ckpt_path = save_checkpoint(
            model, optimizer, epoch, global_step, history, args.checkpoint_dir
        )
        print(f"  checkpoint → {ckpt_path}")

    # 儲存最終 Adapter 權重與訓練歷史
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(args.output, dict(model.adapter.parameters()))

    history_path = Path(args.output).parent / "loss_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nAdapter 權重已儲存至 {args.output}")
    print(f"訓練歷史已儲存至 {history_path}")


if __name__ == "__main__":
    main()

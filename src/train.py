"""
Phase 3: 訓練迴圈建構 (Training Loop & Fine-tuning)

規則：
    - 只針對第二次 Pass 輸出中、對應原始序列位置的 logits 計算 Cross-Entropy Loss
    - 第一次 Pass 不計算任何 Loss
    - 使用 mlx.nn.value_and_grad，梯度只更新 Adapter（base_model 需預先 freeze）
    - 從 batch size = 1 開始測試記憶體水位（24GB M2 Air，避免觸發 swap）

執行方式：
    python src/train.py --model-path google/gemma-2-2b --dataset data/train.jsonl --epochs 3 --batch-size 1
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from .model_wrapper import ReflectiveGemma, ReflectiveGemmaConfig


def loss_fn(model: ReflectiveGemma, input_ids: mx.array, targets: mx.array) -> mx.array:
    """只計算第二次 pass 中、對應原始序列位置的 cross entropy loss。

    第二次 pass 的 logits 形狀為 (batch, N_v + seq_len, vocab_size)。
    前 N_v 個位置是虛擬代幣輸出，不應計入 loss。
    只取後 seq_len 個位置，並做 next-token prediction 的偏移（logits[t] 預測 targets[t+1]）。

    Args:
        input_ids: (batch, seq_len) — 輸入 token ids
        targets:   (batch, seq_len) — 目標 token ids（語言模型訓練時與 input_ids 相同）
    """
    n_v = model.config.num_virtual_tokens
    logits = model(input_ids)       # (batch, n_v + seq_len, vocab_size)

    # 裁掉虛擬代幣部分，並做 next-token 偏移
    # logits[:, n_v:-1, :] 對應預測 targets[:, 1:]
    aligned_logits = logits[:, n_v:-1, :]  # (batch, seq_len-1, vocab_size)
    aligned_targets = targets[:, 1:]        # (batch, seq_len-1)

    return nn.losses.cross_entropy(aligned_logits, aligned_targets, reduction="mean")


def train_one_epoch(model: ReflectiveGemma, optimizer, batches, epoch: int) -> float:
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    running_loss = 0.0
    num_steps = 0
    for num_steps, (input_ids, targets) in enumerate(batches, start=1):
        loss, grads = loss_and_grad_fn(model, input_ids, targets)

        # nn.value_and_grad 只對 trainable_parameters（Adapter）求梯度
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        running_loss += loss.item()
        if (num_steps - 1) % 10 == 0:
            peak_mb = mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx, "metal") else -1
            print(f"epoch {epoch} step {num_steps - 1}: loss={loss.item():.4f}  peak_mem_mb={peak_mb:.1f}")

    return running_loss / max(num_steps, 1)


def build_batches(dataset_path: str, tokenizer, batch_size: int):
    """載入 JSONL 並依 batch_size 切批，回傳 (input_ids, targets) 的 list。"""
    examples = []
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = f"{obj['prompt']} {obj.get('target', '')}"
            examples.append(tokenizer.encode(text))

    batches = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        max_len = max(len(ids) for ids in chunk)
        pad_id = getattr(tokenizer, "pad_id", 0)
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in chunk]
        arr = mx.array(padded)       # (batch, seq_len)
        batches.append((arr, arr))   # targets == input_ids for causal LM

    return batches


def main():
    parser = argparse.ArgumentParser(description="Phase 3: 訓練 Adapter")
    parser.add_argument("--model-path", default="google/gemma-2-2b")
    parser.add_argument("--dataset", default="data/train.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="results/adapter_weights.safetensors")
    args = parser.parse_args()

    from mlx_lm import load
    base_model, tokenizer = load(args.model_path)

    # 凍結 Gemma 主體，只讓 Adapter 可訓練
    base_model.freeze()

    hidden_size = base_model.model.embed_tokens.weight.shape[-1]
    config = ReflectiveGemmaConfig(hidden_size=hidden_size)
    model = ReflectiveGemma(base_model, config)

    optimizer = optim.Adam(learning_rate=args.lr)
    batches = build_batches(args.dataset, tokenizer, args.batch_size)

    loss_history = []
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, optimizer, batches, epoch)
        loss_history.append(avg_loss)
        print(f"=== Epoch {epoch} 平均 Loss: {avg_loss:.4f} ===")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(args.output, dict(model.adapter.parameters()))
    with open("results/loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)
    print(f"Adapter 權重已儲存至 {args.output}")


if __name__ == "__main__":
    main()

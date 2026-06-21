"""
Phase 3: 訓練迴圈建構 (Training Loop & Fine-tuning)

規則：
    - 只針對第二次 Pass 輸出的 logits 計算 Cross-Entropy Loss
    - 第一次 Pass 不計算任何 Loss
    - 使用 mlx.core.value_and_grad，梯度只更新 Adapter 的參數
    - 從 batch size = 1 開始測試記憶體水位（24GB M2 Air，避免觸發 swap）

執行方式：
    python src/train.py --model-path google/gemma-2-2b --dataset data/train.jsonl --epochs 3 --batch-size 1
"""

import argparse
import json

import mlx.core as mx
import mlx.optimizers as optim

from .model_wrapper import ReflectiveGemma, ReflectiveGemmaConfig


def loss_fn(model: ReflectiveGemma, input_ids: mx.array, targets: mx.array) -> mx.array:
    """只計算第二次 pass 的 cross entropy loss。"""
    logits = model(input_ids)
    # TODO: 對齊 logits 與 targets 的維度，計算 cross entropy
    # return mx.mean(nn.losses.cross_entropy(logits, targets))
    raise NotImplementedError("TODO: 實作 cross entropy 計算")


def train_one_epoch(model: ReflectiveGemma, optimizer, batches, epoch: int):
    loss_and_grad_fn = mx.value_and_grad(loss_fn)

    running_loss = 0.0
    for step, (input_ids, targets) in enumerate(batches):
        loss, grads = loss_and_grad_fn(model, input_ids, targets)

        # 只更新 Adapter 的梯度（grads 結構需對應 model.trainable_parameters()）
        optimizer.update(model.adapter, grads["adapter"])
        mx.eval(model.adapter.parameters(), optimizer.state)

        running_loss += loss.item()
        if step % 10 == 0:
            peak_mb = mx.metal.get_peak_memory() / (1024 * 1024) if hasattr(mx, "metal") else -1
            print(f"epoch {epoch} step {step}: loss={loss.item():.4f} peak_mem_mb={peak_mb:.1f}")

    return running_loss / max(step + 1, 1)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: 訓練 Adapter")
    parser.add_argument("--model-path", default="google/gemma-2-2b")
    parser.add_argument("--dataset", default="data/train.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="results/adapter_weights.npz")
    args = parser.parse_args()

    # TODO: 載入 base_model / tokenizer，建立 ReflectiveGemma
    # base_model, tokenizer = load(args.model_path)
    # config = ReflectiveGemmaConfig(hidden_size=base_model.config.hidden_size)
    # model = ReflectiveGemma(base_model, config)

    # optimizer = optim.Adam(learning_rate=args.lr)

    # TODO: 載入資料、依 batch_size 切批
    # batches = build_batches(args.dataset, tokenizer, args.batch_size)

    # loss_history = []
    # for epoch in range(args.epochs):
    #     avg_loss = train_one_epoch(model, optimizer, batches, epoch)
    #     loss_history.append(avg_loss)
    #     print(f"=== Epoch {epoch} 平均 Loss: {avg_loss:.4f} ===")

    # mx.save_safetensors(args.output, model.adapter.parameters())
    # with open("results/loss_history.json", "w") as f:
    #     json.dump(loss_history, f, indent=2)

    raise NotImplementedError("TODO: 串接上方流程後移除此行")


if __name__ == "__main__":
    main()

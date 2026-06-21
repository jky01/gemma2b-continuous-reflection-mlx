"""
Checkpoint 儲存與讀取工具

提供三個公開函式：
    save_checkpoint   — 儲存 Adapter 權重 + Optimizer 狀態 + Metadata
    find_latest_checkpoint — 找出最新 checkpoint 路徑（找不到回傳 None）
    load_checkpoint   — 從 checkpoint 恢復模型與 optimizer 狀態

Checkpoint 目錄結構：
    checkpoint_dir/
    ├── latest.txt                 ← 指向最新 epoch 的名稱
    ├── epoch_00/
    │   ├── adapter.safetensors    ← Adapter 參數（可單獨用來推論）
    │   ├── optimizer.safetensors  ← Adam 的 step / m / v
    │   └── meta.json             ← epoch / global_step / loss history
    ├── epoch_01/
    │   └── ...
    └── ...
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


# ─── 儲存 ──────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    global_step: int,
    history: list,
    checkpoint_dir: str,
) -> Path:
    """儲存一個完整的訓練狀態 checkpoint。

    Args:
        model:          ReflectiveGemma 模型（只存 adapter 部分）
        optimizer:      MLX Adam optimizer
        epoch:          剛完成的 epoch 索引（從 0 開始）
        global_step:    全域訓練步數（跨 epoch 累計）
        history:        list of dict，每個 epoch 的訓練紀錄
        checkpoint_dir: checkpoint 根目錄路徑

    Returns:
        本次 checkpoint 的目錄 Path
    """
    ckpt_root = Path(checkpoint_dir)
    epoch_dir = ckpt_root / f"epoch_{epoch:02d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # 1. Adapter 參數（flat safetensors）
    adapter_flat = dict(tree_flatten(model.adapter.parameters()))
    mx.save_safetensors(str(epoch_dir / "adapter.safetensors"), adapter_flat)

    # 2. Optimizer 狀態：step + learning_rate + 每個參數的 m/v
    #    tree_flatten 只保留葉節點（mx.array），空 dict（如 GELU）會被跳過，
    #    restore 時由 optimizer.init() 自動補回空 dict。
    opt_flat = dict(tree_flatten(optimizer.state))
    if opt_flat:
        mx.save_safetensors(str(epoch_dir / "optimizer.safetensors"), opt_flat)

    # 3. Metadata
    meta = {
        "epoch":       epoch,
        "global_step": global_step,
        "history":     history,
    }
    with open(epoch_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 4. 更新 latest 指標
    (ckpt_root / "latest.txt").write_text(f"epoch_{epoch:02d}")

    return epoch_dir


# ─── 尋找 ─────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[Path]:
    """尋找最新的有效 checkpoint，回傳其目錄 Path；找不到則回傳 None。

    有效條件：latest.txt 存在 + 對應目錄存在 + meta.json 存在。
    """
    ckpt_root   = Path(checkpoint_dir)
    latest_file = ckpt_root / "latest.txt"

    if not latest_file.exists():
        return None

    ckpt_name = latest_file.read_text().strip()
    ckpt_path = ckpt_root / ckpt_name

    if not (ckpt_path / "meta.json").exists():
        return None

    return ckpt_path


# ─── 讀取 ─────────────────────────────────────────────────────────────────────

def load_checkpoint(
    model,
    optimizer,
    checkpoint_dir: str,
) -> Tuple[int, int, List[dict]]:
    """從最新 checkpoint 恢復訓練狀態。

    Returns:
        (start_epoch, global_step, history)
        — start_epoch:  下一輪要從哪個 epoch 開始（= 儲存的 epoch + 1）
        — global_step:  截至上次的全域步數（讓 LR schedule 接續）
        — history:      之前所有 epoch 的訓練紀錄

    若找不到 checkpoint，回傳 (0, 0, []) 並從頭開始。
    """
    ckpt_path = find_latest_checkpoint(checkpoint_dir)

    if ckpt_path is None:
        print("找不到 checkpoint，從頭開始訓練。")
        return 0, 0, []

    # — Adapter 參數 —
    adapter_file = ckpt_path / "adapter.safetensors"
    if adapter_file.exists():
        weights = mx.load(str(adapter_file))
        model.adapter.load_weights(list(weights.items()))
        mx.eval(model.adapter.parameters())

    # — Optimizer 狀態 —
    opt_file = ckpt_path / "optimizer.safetensors"
    if opt_file.exists():
        opt_flat    = dict(mx.load(str(opt_file)))
        # tree_unflatten 根據 dotted key 重建巢狀結構
        opt_state   = tree_unflatten(list(opt_flat.items()))
        optimizer.state = opt_state
        # 確保 mx.eval 讓 m/v/step 完成計算
        mx.eval(optimizer.state)

    # — Metadata —
    with open(ckpt_path / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    start_epoch = meta["epoch"] + 1
    global_step = meta.get("global_step", 0)
    history     = meta.get("history", [])

    saved_epoch = meta["epoch"]
    print(
        f"從 checkpoint 恢復：\n"
        f"  路徑        = {ckpt_path}\n"
        f"  已完成 epoch = {saved_epoch}\n"
        f"  global_step  = {global_step}\n"
        f"  繼續 epoch   ≥ {start_epoch}"
    )
    return start_epoch, global_step, history

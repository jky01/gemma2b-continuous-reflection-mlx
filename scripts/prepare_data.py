"""
訓練資料準備腳本

從 HuggingFace 下載資料集並轉換成專案用的 JSONL 格式。

支援來源：
    gsm8k   — 8500 筆英文小學數學推理題（OpenAI / HuggingFace）
    belle   — 25 萬筆中文數學題（BelleGroup）

使用方式：
    # 下載 GSM8K，取 500 筆訓練 / 100 筆測試
    python scripts/prepare_data.py --source gsm8k --train 500 --test 100

    # 下載中文數學題
    python scripts/prepare_data.py --source belle --train 500 --test 100

    # 指定輸出目錄
    python scripts/prepare_data.py --source gsm8k --out-dir data/gsm8k
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ── 來源處理器 ────────────────────────────────────────────────────────────────

def _extract_gsm8k_answer(answer_text: str) -> str:
    """從 GSM8K 答案欄位取出最終數字（#### 後面的部分）。"""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    # fallback：取最後一行的數字
    for line in reversed(answer_text.strip().splitlines()):
        nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", line.replace(",", ""))
        if nums:
            return nums[-1]
    return answer_text.strip()


def load_gsm8k(n_train: int, n_test: int):
    """載入 GSM8K 並回傳 (train_examples, test_examples)。"""
    from datasets import load_dataset
    print("下載 GSM8K …")
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test  = load_dataset("openai/gsm8k", "main", split="test")

    def convert(ex):
        return {
            "prompt": ex["question"].strip(),
            "target": _extract_gsm8k_answer(ex["answer"]),
        }

    train = [convert(ex) for ex in ds_train.select(range(min(n_train, len(ds_train))))]
    test  = [convert(ex) for ex in ds_test.select( range(min(n_test,  len(ds_test))))]
    return train, test


def load_belle(n_train: int, n_test: int):
    """載入 BelleGroup 中文數學題並回傳 (train_examples, test_examples)。"""
    from datasets import load_dataset
    print("下載 BelleGroup/school_math_0.25M …")
    ds = load_dataset("BelleGroup/school_math_0.25M", split="train")

    examples = []
    for ex in ds:
        instruction = ex.get("instruction", "").strip()
        output      = ex.get("output", "").strip()
        if not instruction or not output:
            continue
        # 嘗試從輸出末尾提取最終答案（數字）
        nums = re.findall(r"-?\d+(?:\.\d+)?", output.replace(",", ""))
        target = nums[-1] if nums else output[:50]
        examples.append({"prompt": instruction, "target": target})
        if len(examples) >= n_train + n_test:
            break

    train = examples[:n_train]
    test  = examples[n_train: n_train + n_test]
    return train, test


# ── 寫出 JSONL ────────────────────────────────────────────────────────────────

def write_jsonl(examples: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  寫出 {len(examples):>5} 筆 → {path}")


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="下載並準備訓練資料")
    parser.add_argument("--source",  choices=["gsm8k", "belle"], default="gsm8k",
                        help="資料來源（預設 gsm8k）")
    parser.add_argument("--train",   type=int, default=500,  help="訓練集筆數")
    parser.add_argument("--test",    type=int, default=100,  help="測試集筆數")
    parser.add_argument("--out-dir", default="data",         help="輸出目錄")
    args = parser.parse_args()

    out = Path(args.out_dir)

    if args.source == "gsm8k":
        train, test = load_gsm8k(args.train, args.test)
    else:
        train, test = load_belle(args.train, args.test)

    write_jsonl(train, out / "train.jsonl")
    write_jsonl(test,  out / "test.jsonl")

    print(f"\n完成！訓練：{len(train)} 筆，測試：{len(test)} 筆")
    print(f"範例（第一筆）：")
    print(f"  prompt : {train[0]['prompt'][:80]}")
    print(f"  target : {train[0]['target']}")


if __name__ == "__main__":
    main()

"""
Phase 4: 評估驗證與推論優化 (Evaluation & Optimization)

任務：
    1. 對比測試：原始單向生成 vs. 加入 Adapter 的雙向生成（準確率與生成品質）。
    2. 分析特徵空間：將 Adapter 生成的虛擬代幣向量，與真實單字 Embedding 計算
       Cosine Similarity，嘗試解讀模型「反思」了什麼內容。
    3. 封裝乾淨、獨立的推論 Class，方便未來匯入使用。

執行方式：
    python src/evaluate.py --model-path google/gemma-2-2b \
        --adapter-weights results/adapter_weights.npz \
        --dataset data/test.jsonl
"""

import argparse
import json

import mlx.core as mx

from .model_wrapper import ReflectiveGemma, ReflectiveGemmaConfig


def cosine_similarity(a: mx.array, b: mx.array) -> mx.array:
    a_norm = a / mx.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / mx.linalg.norm(b, axis=-1, keepdims=True)
    return mx.sum(a_norm * b_norm, axis=-1)


def compare_baseline_vs_reflective(model: ReflectiveGemma, base_model, examples):
    """對比原始單向生成與雙向（反思）生成的準確率。

    TODO:
        - 對每個 example 分別用 base_model（單向）與 model（雙向）生成
        - 比較與 ground truth 的吻合度（exact match / token-level accuracy）
    """
    results = {"baseline_accuracy": None, "reflective_accuracy": None}
    return results


def analyze_virtual_tokens(model: ReflectiveGemma, embedding_table: mx.array, examples):
    """印出虛擬代幣與真實字詞 embedding 的 cosine similarity，找出最相近的字詞。

    TODO:
        - 取得每個 example 的 p_soft（4 個虛擬代幣向量）
        - 對 embedding_table 做相似度排序，找出 top-k 最相近的真實字詞
    """
    findings = []
    return findings


class ReflectiveGemmaInference:
    """封裝雙次傳遞推論邏輯，供未來直接 import 使用。"""

    def __init__(self, model: ReflectiveGemma, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        """TODO: 實作完整生成流程（含 reflection pass）。"""
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Phase 4: 評估與分析")
    parser.add_argument("--model-path", default="google/gemma-2-2b")
    parser.add_argument("--adapter-weights", default="results/adapter_weights.npz")
    parser.add_argument("--dataset", default="data/test.jsonl")
    parser.add_argument("--report-out", default="results/evaluation_report.json")
    args = parser.parse_args()

    # TODO: 載入模型、Adapter 權重、測試資料，串接以上函式
    report = {
        "comparison": None,
        "virtual_token_analysis": None,
    }

    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"報告已寫入 {args.report_out}")


if __name__ == "__main__":
    main()

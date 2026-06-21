"""
Phase 4: 評估驗證與推論優化 (Evaluation & Optimization)

任務：
    1. 對比測試：原始單向生成 vs. 加入 Adapter 的雙向生成（準確率與生成品質）。
    2. 分析特徵空間：將 Adapter 生成的虛擬代幣向量，與真實單字 Embedding 計算
       Cosine Similarity，嘗試解讀模型「反思」了什麼內容。
    3. 封裝乾淨、獨立的推論 Class，方便未來匯入使用。

執行方式：
    python src/evaluate.py --model-path google/gemma-2-2b \
        --adapter-weights results/adapter_weights.safetensors \
        --dataset data/sample_qa.jsonl
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from .model_wrapper import ReflectiveGemma, ReflectiveGemmaConfig


def cosine_similarity(a: mx.array, b: mx.array) -> mx.array:
    """計算 a 與 b 各行的 cosine similarity。

    Args:
        a: shape (..., D)
        b: shape (..., D)  或可與 a broadcast 的形狀

    Returns:
        shape (...,)  各位置的相似度，值域 [-1, 1]
    """
    a_norm = a / (mx.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (mx.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return mx.sum(a_norm * b_norm, axis=-1)


class ReflectiveGemmaInference:
    """封裝雙次傳遞推論邏輯，供未來直接 import 使用。"""

    def __init__(self, model: ReflectiveGemma, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        """自迴歸生成，每一步都執行完整的反思迴圈（雙次傳遞）。

        生成流程：
            1. 以當前序列執行 ReflectiveGemma forward pass
            2. 取第二次 pass 輸出的最後位置 logits（即原始序列最後一個 token 的預測）
            3. Argmax 取下一個 token，追加到序列
            4. 重複直到 EOS 或達到 max_tokens
        """
        input_ids = mx.array(self.tokenizer.encode(prompt))[None]  # (1, seq_len)

        for _ in range(max_tokens):
            n_v = self.model.config.num_virtual_tokens
            logits = self.model(input_ids)          # (1, n_v + cur_len, vocab_size)

            # 取原始序列最後一個位置的預測（即 logits 的最後一個位置）
            next_token_logits = logits[0, -1, :]    # (vocab_size,)
            next_id = int(mx.argmax(next_token_logits).item())
            mx.eval(next_id)

            eos_id = getattr(self.tokenizer, "eos_token_id",
                             getattr(self.tokenizer, "eos_id", None))
            if eos_id is not None and next_id == eos_id:
                break

            input_ids = mx.concatenate(
                [input_ids, mx.array([[next_id]])], axis=1
            )

        all_ids = input_ids[0].tolist()
        prompt_len = len(self.tokenizer.encode(prompt))
        return self.tokenizer.decode(all_ids[prompt_len:])


def compare_baseline_vs_reflective(model: ReflectiveGemma, base_model, tokenizer, examples):
    """對比原始單向生成與雙向（反思）生成的準確率（exact-match in output）。"""
    from mlx_lm import generate as mlx_generate

    baseline_correct = 0
    reflective_correct = 0
    inference = ReflectiveGemmaInference(model, tokenizer)

    for ex in examples:
        prompt = ex["prompt"]
        target = str(ex.get("target", "")).strip()

        baseline_out = mlx_generate(base_model, tokenizer, prompt=prompt, max_tokens=32, verbose=False)
        if target.lower() in baseline_out.lower():
            baseline_correct += 1

        reflective_out = inference.generate(prompt, max_tokens=32)
        if target.lower() in reflective_out.lower():
            reflective_correct += 1

    total = len(examples)
    return {
        "total": total,
        "baseline_correct": baseline_correct,
        "reflective_correct": reflective_correct,
        "baseline_accuracy": baseline_correct / max(total, 1),
        "reflective_accuracy": reflective_correct / max(total, 1),
    }


def analyze_virtual_tokens(model: ReflectiveGemma, tokenizer, examples, top_k: int = 5):
    """印出虛擬代幣與真實字詞 embedding 的 cosine similarity，解讀反思內容。

    embedding_table: model.base_model.model.embed_tokens.weight
        shape (vocab_size, hidden_size)
    p_soft[i]:  shape (hidden_size,) — 第 i 個虛擬代幣
    """
    embedding_table = model.base_model.model.embed_tokens.weight  # (vocab_size, hidden_size)
    findings = []

    for ex in examples:
        input_ids = mx.array(tokenizer.encode(ex["prompt"]))[None]  # (1, seq_len)

        h_top = model.first_pass_hidden_state(input_ids)  # (1, hidden_size)
        p_soft = model.adapter(h_top)                      # (1, num_virtual_tokens, hidden_size)
        mx.eval(p_soft)

        token_findings = []
        for i in range(p_soft.shape[1]):
            v_tok = p_soft[0, i, :]   # (hidden_size,)

            # cosine sim: v_tok vs. every row of embedding_table
            sims = cosine_similarity(v_tok[None], embedding_table)  # (vocab_size,)
            mx.eval(sims)

            top_indices = mx.argsort(-sims)[:top_k].tolist()
            top_words = [tokenizer.decode([idx]) for idx in top_indices]
            top_scores = [float(sims[idx].item()) for idx in top_indices]

            token_findings.append({
                "virtual_token_idx": i,
                "top_k_words": top_words,
                "cosine_scores": [round(s, 4) for s in top_scores],
            })

        findings.append({
            "prompt": ex["prompt"],
            "virtual_token_analysis": token_findings,
        })

    return findings


def main():
    parser = argparse.ArgumentParser(description="Phase 4: 評估與分析")
    parser.add_argument("--model-path", default="google/gemma-2-2b")
    parser.add_argument("--adapter-weights", default="results/adapter_weights.safetensors")
    parser.add_argument("--dataset", default="data/sample_qa.jsonl")
    parser.add_argument("--report-out", default="results/evaluation_report.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    from mlx_lm import load
    base_model, tokenizer = load(args.model_path)
    base_model.freeze()

    hidden_size = base_model.model.embed_tokens.weight.shape[-1]
    config = ReflectiveGemmaConfig(hidden_size=hidden_size)
    model = ReflectiveGemma(base_model, config)

    # 載入訓練好的 Adapter 權重
    weights = mx.load(args.adapter_weights)
    model.adapter.load_weights(list(weights.items()))

    with Path(args.dataset).open("r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    comparison = compare_baseline_vs_reflective(model, base_model, tokenizer, examples)
    vt_analysis = analyze_virtual_tokens(model, tokenizer, examples, top_k=args.top_k)

    report = {
        "comparison": comparison,
        "virtual_token_analysis": vt_analysis,
    }

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"報告已寫入 {args.report_out}")
    print(f"Baseline 準確率: {comparison['baseline_accuracy']:.2%}")
    print(f"反思模型準確率: {comparison['reflective_accuracy']:.2%}")


if __name__ == "__main__":
    main()

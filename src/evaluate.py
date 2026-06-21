"""
Phase 4: 評估驗證與推論優化 (Evaluation & Optimization)

任務：
    1. 對比測試：原始單向生成 vs. 加入 Adapter 的雙向生成（準確率與生成品質）。
    2. 分析特徵空間：將 Adapter 生成的虛擬代幣向量，與真實單字 Embedding 計算
       Cosine Similarity，嘗試解讀模型「反思」了什麼內容。
    3. 封裝乾淨、獨立的推論 Class，方便未來匯入使用。

執行方式：
    python -m src.evaluate \
        --model-path google/gemma-2-2b \
        --adapter-weights results/adapter_weights.safetensors \
        --dataset data/test.jsonl \
        --temperature 0.7 --top-p 0.9
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .data_utils import compute_perplexity, load_jsonl, next_token_loss, tokenize_example
from .model_wrapper import ReflectiveGemma, ReflectiveGemmaConfig


# ─── 工具函式 ──────────────────────────────────────────────────────────────────

def cosine_similarity(a: mx.array, b: mx.array) -> mx.array:
    """計算 a 與 b 各行的 cosine similarity，支援 broadcast。

    Args:
        a: shape (..., D)
        b: shape (..., D)

    Returns:
        shape (...,) 各位置相似度，值域 [-1, 1]
    """
    a_norm = a / (mx.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (mx.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return mx.sum(a_norm * b_norm, axis=-1)


def sample_token(logits: mx.array, temperature: float = 0.0, top_p: float = 1.0) -> int:
    """從 logits 取樣下一個 token。

    Args:
        logits:      shape (vocab_size,) — 未正規化的 log-odds
        temperature: 0.0 → greedy argmax；> 0 → 機率取樣
        top_p:       nucleus sampling 累積機率門檻（< 1.0 時啟用）

    Returns:
        取樣到的 token id（int）
    """
    if temperature == 0.0:
        return int(mx.argmax(logits).item())

    scaled = logits / temperature

    if top_p < 1.0:
        # Nucleus (top-p) sampling
        sorted_idx    = mx.argsort(-scaled)          # 從大到小排序
        sorted_logits = scaled[sorted_idx]
        probs         = mx.softmax(sorted_logits)
        cum_probs     = mx.cumsum(probs)

        # 保留累積機率超過 top_p 之前的 token（至少保留機率最大的一個）
        mask          = (cum_probs - probs) < top_p
        sorted_logits = mx.where(mask, sorted_logits, mx.array(float("-inf")))

        # 還原原始順序
        inv_idx = mx.argsort(sorted_idx)
        scaled  = sorted_logits[inv_idx]

    return int(mx.random.categorical(scaled[None]).item())


# ─── 推論類別 ──────────────────────────────────────────────────────────────────

class ReflectiveGemmaInference:
    """封裝雙次傳遞推論邏輯，供未來直接 import 使用。"""

    def __init__(self, model: ReflectiveGemma, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """自迴歸生成，每一步都執行完整的反思迴圈（雙次傳遞）。

        Args:
            prompt:      輸入提示詞
            max_tokens:  最多生成幾個 token
            temperature: 0.0 → greedy；> 0 → 機率取樣
            top_p:       nucleus sampling 門檻（temperature > 0 時有效）
        """
        eos_id = getattr(
            self.tokenizer, "eos_token_id",
            getattr(self.tokenizer, "eos_id", None),
        )

        input_ids = mx.array(self.tokenizer.encode(prompt))[None]  # (1, seq_len)

        for _ in range(max_tokens):
            logits = self.model(input_ids)     # (1, N_v + cur_len, vocab_size)

            # 取最後一個位置的預測（原始序列末端）
            next_token_logits = logits[0, -1, :]   # (vocab_size,)
            mx.eval(next_token_logits)

            next_id = sample_token(next_token_logits, temperature, top_p)

            if eos_id is not None and next_id == eos_id:
                break

            input_ids = mx.concatenate(
                [input_ids, mx.array([[next_id]])], axis=1
            )

        all_ids   = input_ids[0].tolist()
        prompt_len = len(self.tokenizer.encode(prompt))
        return self.tokenizer.decode(all_ids[prompt_len:])


# ─── 評估函式 ──────────────────────────────────────────────────────────────────

def compare_baseline_vs_reflective(
    model: ReflectiveGemma,
    base_model,
    tokenizer,
    examples,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 32,
):
    """對比原始單向生成與雙向（反思）生成的準確率（含 target 的 exact-match）。"""
    from mlx_lm import generate as mlx_generate

    baseline_correct   = 0
    reflective_correct = 0
    inference = ReflectiveGemmaInference(model, tokenizer)

    for ex in examples:
        prompt = ex["prompt"]
        target = str(ex.get("target", "")).strip()

        baseline_out = mlx_generate(
            base_model, tokenizer,
            prompt=prompt, max_tokens=max_tokens, verbose=False,
        )
        if target.lower() in baseline_out.lower():
            baseline_correct += 1

        reflective_out = inference.generate(
            prompt, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p,
        )
        if target.lower() in reflective_out.lower():
            reflective_correct += 1

    total = len(examples)
    return {
        "total":                total,
        "baseline_correct":     baseline_correct,
        "reflective_correct":   reflective_correct,
        "baseline_accuracy":    baseline_correct  / max(total, 1),
        "reflective_accuracy":  reflective_correct / max(total, 1),
    }


def compute_reflective_perplexity(
    model: ReflectiveGemma,
    tokenizer,
    examples,
    max_length: int = 256,
) -> dict:
    """計算反思模型在測試集上的 perplexity（使用 loss_fn 的偏移邏輯）。"""
    from .train import loss_fn

    total_loss = 0.0
    num_steps  = 0
    for ex in examples:
        ids = tokenize_example(tokenizer, ex["prompt"], ex.get("target", ""), max_length)
        if len(ids) < 2:
            continue
        input_ids = mx.array(ids)[None]
        loss = loss_fn(model, input_ids, input_ids)
        mx.eval(loss)
        total_loss += loss.item()
        num_steps  += 1

    avg_loss = total_loss / max(num_steps, 1)
    return {
        "avg_loss":   round(avg_loss, 6),
        "perplexity": round(compute_perplexity(avg_loss), 4),
        "n_examples": num_steps,
    }


def analyze_virtual_tokens(
    model: ReflectiveGemma,
    tokenizer,
    examples,
    top_k: int = 5,
) -> list:
    """印出虛擬代幣與真實字詞 embedding 的 cosine similarity，解讀反思內容。"""
    embedding_table = model.base_model.model.embed_tokens.weight  # (vocab_size, hidden_size)
    findings = []

    for ex in examples:
        ids      = tokenizer.encode(ex["prompt"])
        input_ids = mx.array(ids)[None]

        h_top  = model.first_pass_hidden_state(input_ids)  # (1, hidden_size)
        p_soft = model.adapter(h_top)                       # (1, N_v, hidden_size)
        mx.eval(p_soft)

        token_findings = []
        for i in range(p_soft.shape[1]):
            v_tok = p_soft[0, i, :]                                     # (hidden_size,)
            sims  = cosine_similarity(v_tok[None], embedding_table)     # (vocab_size,)
            mx.eval(sims)

            top_idx   = mx.argsort(-sims)[:top_k].tolist()
            top_words = [tokenizer.decode([idx]) for idx in top_idx]
            top_scores = [round(float(sims[idx].item()), 4) for idx in top_idx]

            token_findings.append({
                "virtual_token_idx": i,
                "top_k_words":       top_words,
                "cosine_scores":     top_scores,
            })

        findings.append({"prompt": ex["prompt"], "virtual_token_analysis": token_findings})

    return findings


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 4: 評估與分析")
    parser.add_argument("--model-path",      default="google/gemma-2-2b")
    parser.add_argument("--adapter-weights", default="results/adapter_weights.safetensors")
    parser.add_argument("--dataset",         default="data/test.jsonl")
    parser.add_argument("--report-out",      default="results/evaluation_report.json")
    parser.add_argument("--top-k",           type=int,   default=5)
    parser.add_argument("--temperature",     type=float, default=0.0,
                        help="0.0 → greedy；> 0 → 機率取樣")
    parser.add_argument("--top-p",           type=float, default=1.0,
                        help="Nucleus sampling 門檻（temperature > 0 時有效）")
    parser.add_argument("--max-tokens",      type=int,   default=32)
    parser.add_argument("--max-length",      type=int,   default=256)
    args = parser.parse_args()

    from mlx_lm import load
    base_model, tokenizer = load(args.model_path)
    base_model.freeze()

    hidden_size = base_model.model.embed_tokens.weight.shape[-1]
    config = ReflectiveGemmaConfig(hidden_size=hidden_size)
    model  = ReflectiveGemma(base_model, config)

    weights = mx.load(args.adapter_weights)
    model.adapter.load_weights(list(weights.items()))

    examples = load_jsonl(args.dataset)

    comparison  = compare_baseline_vs_reflective(
        model, base_model, tokenizer, examples,
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
    )
    ppl_metrics = compute_reflective_perplexity(model, tokenizer, examples, args.max_length)
    vt_analysis = analyze_virtual_tokens(model, tokenizer, examples, top_k=args.top_k)

    report = {
        "generation_comparison": comparison,
        "reflective_perplexity": ppl_metrics,
        "virtual_token_analysis": vt_analysis,
    }

    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"報告已寫入 {args.report_out}")
    print(f"Baseline    準確率: {comparison['baseline_accuracy']:.2%}")
    print(f"反思模型準確率: {comparison['reflective_accuracy']:.2%}")
    print(f"反思模型 Perplexity: {ppl_metrics['perplexity']:.4f}")


if __name__ == "__main__":
    main()

"""
Phase 2: 覆寫 Forward Pass — 雙次傳遞與拼接邏輯 (Architecture Modification)

數學流程：
    H_top      = Gemma(X_input)[-1]                      # 第一次傳遞，取高階特徵
    P_soft     = Adapter(H_top)                           # 轉譯成虛擬代幣
    X_second   = Concat(P_soft, Embedding(X_input))        # 拼接
    Y_pred     = Gemma(X_second)                           # 第二次傳遞，重新預測

本檔案負責把 Gemma 主體（凍結）與 ReflectionAdapter（可訓練）接起來，
成為一個可以直接呼叫的 ReflectiveGemma 類別。
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .adapter import ReflectionAdapter


@dataclass
class ReflectiveGemmaConfig:
    hidden_size: int
    num_virtual_tokens: int = 4
    adapter_expansion: int = 4


class ReflectiveGemma(nn.Module):
    """包裹原始 Gemma 模型，加入一次內部「反思迴圈」。"""

    def __init__(self, base_model, config: ReflectiveGemmaConfig):
        super().__init__()
        self.base_model = base_model  # 凍結的 Gemma 2B 主體
        self.config = config
        self.adapter = ReflectionAdapter(
            hidden_size=config.hidden_size,
            num_virtual_tokens=config.num_virtual_tokens,
            expansion=config.adapter_expansion,
        )

    def first_pass_hidden_state(self, input_ids: mx.array) -> mx.array:
        """取得第一次傳遞、最後一層、最後一個 token 的 hidden state。

        TODO:
            - 需要呼叫 base_model 並要求回傳 hidden_states（而非僅 logits）。
            - 多數 mlx-lm 模型預設只回傳 logits，需要修改/包裝其
              forward 方法，讓最後一層輸出（pre-lm-head）也能取得。
        """
        raise NotImplementedError("TODO: 攔截並回傳 H_top, shape (batch, hidden_size)")

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        """取得輸入字串的 token embedding，用於和 P_soft 拼接。"""
        raise NotImplementedError("TODO: 回傳 shape (batch, seq_len, hidden_size)")

    def __call__(self, input_ids: mx.array) -> mx.array:
        # 第一次傳遞
        h_top = self.first_pass_hidden_state(input_ids)

        # Adapter 轉譯
        p_soft = self.adapter(h_top)  # (batch, num_virtual_tokens, hidden_size)

        # 拼接
        input_embeds = self.embed_tokens(input_ids)  # (batch, seq_len, hidden_size)
        x_second = mx.concatenate([p_soft, input_embeds], axis=1)

        # 第二次傳遞（重新預測）
        y_pred = self.base_model(inputs_embeds=x_second)
        return y_pred

    def trainable_parameters(self):
        """只回傳 Adapter 的參數，Gemma 主體維持凍結。"""
        return self.adapter.parameters()

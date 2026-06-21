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
        """第一次傳遞，取最後一層、最後一個 token 的 hidden state。

        mlx-lm 的 model.model（GemmaModel）回傳 (hidden_states, cache)，
        hidden_states.shape = (batch, seq_len, hidden_size)。

        Returns:
            shape (batch, hidden_size)
        """
        hidden_states, _ = self.base_model.model(input_ids)
        return hidden_states[:, -1, :]  # 最後一個 token 的高階特徵

    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        """取得 token embedding，供第二次傳遞拼接用。

        Returns:
            shape (batch, seq_len, hidden_size)
        """
        return self.base_model.model.embed_tokens(input_ids)

    def _second_pass(self, x_combined: mx.array) -> mx.array:
        """以預先計算的 embedding（含虛擬代幣）執行第二次傳遞。

        使用 mlx-lm >= 0.18.0 的 inputs_embeds 參數直接跳過 embed_tokens，
        讓虛擬代幣與原始 embedding 的拼接向量進入 transformer body。

        Args:
            x_combined: shape (batch, num_virtual_tokens + seq_len, hidden_size)

        Returns:
            logits: shape (batch, num_virtual_tokens + seq_len, vocab_size)
        """
        hidden_states, _ = self.base_model.model(inputs_embeds=x_combined)
        return self.base_model.lm_head(hidden_states)

    def __call__(self, input_ids: mx.array) -> mx.array:
        # 第一次傳遞：提取高階特徵
        h_top = self.first_pass_hidden_state(input_ids)           # (batch, hidden_size)

        # Adapter 轉譯：生成虛擬代幣（soft prompts）
        p_soft = self.adapter(h_top)                               # (batch, N_v, hidden_size)

        # 拼接：虛擬代幣前置於原始 token embedding
        input_embeds = self.embed_tokens(input_ids)                # (batch, seq_len, hidden_size)
        x_second = mx.concatenate([p_soft, input_embeds], axis=1) # (batch, N_v+seq_len, hidden_size)

        # 第二次傳遞：帶前綴脈絡重新預測
        return self._second_pass(x_second)                         # (batch, N_v+seq_len, vocab_size)

    def trainable_parameters(self):
        """只回傳 Adapter 的參數（Gemma 主體由外部 freeze() 凍結）。"""
        return self.adapter.parameters()

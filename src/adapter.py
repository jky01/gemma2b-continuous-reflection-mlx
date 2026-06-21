"""
Phase 2: Adapter 橋接器 (Architecture Modification)

P_soft = Adapter(H_top)

Adapter 是一個兩層線性映射 + GELU 的微型 MLP：
    輸入維度：hidden_size（Gemma 2B 的隱藏層維度）
    輸出維度：4 * hidden_size（代表 4 個虛擬代幣 / soft prompt tokens）
"""

import mlx.core as mx
import mlx.nn as nn


class ReflectionAdapter(nn.Module):
    """將最後一層 hidden state 轉譯為 N 個虛擬代幣（soft prompt）。"""

    def __init__(self, hidden_size: int, num_virtual_tokens: int = 4, expansion: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_virtual_tokens = num_virtual_tokens

        intermediate_size = hidden_size * expansion
        output_size = hidden_size * num_virtual_tokens

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, output_size)

    def __call__(self, h_top: mx.array) -> mx.array:
        """
        Args:
            h_top: shape (batch, hidden_size) — 最後一層、最後一個 token 的 hidden state

        Returns:
            p_soft: shape (batch, num_virtual_tokens, hidden_size)
        """
        x = self.fc1(h_top)
        x = self.act(x)
        x = self.fc2(x)
        batch = h_top.shape[0]
        return x.reshape(batch, self.num_virtual_tokens, self.hidden_size)

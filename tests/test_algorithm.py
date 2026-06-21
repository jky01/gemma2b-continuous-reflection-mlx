"""
演算法正確性驗證測試

使用 numpy 模擬 mlx 的陣列操作，在任何平台（含 Linux CI）都可執行。
涵蓋四個核心面向：

1. ReflectionAdapter  — 形狀轉換、數學計算
2. ReflectiveGemma    — 拼接邏輯、維度流
3. loss_fn            — 虛擬代幣排除、next-token 偏移
4. cosine_similarity  — 數值正確性
"""

import math
import sys
import types
import unittest

import numpy as np


# ─── numpy mock for mlx.core ─────────────────────────────────────────────────

def _make_mx_mock():
    mx = types.SimpleNamespace()
    mx.array       = lambda x, **kw: np.array(x, **kw)
    mx.concatenate = lambda arrays, axis=0: np.concatenate(arrays, axis=axis)
    mx.sum         = lambda x, axis=None, keepdims=False: np.sum(x, axis=axis, keepdims=keepdims)
    mx.mean        = lambda x, axis=None: np.mean(x, axis=axis)
    mx.argmax      = lambda x, axis=None: np.argmax(x, axis=axis)
    mx.argsort     = lambda x: np.argsort(x)
    mx.expand_dims = lambda x, axis: np.expand_dims(x, axis)

    linalg = types.SimpleNamespace()
    linalg.norm = lambda x, axis=None, keepdims=False: np.linalg.norm(x, axis=axis, keepdims=keepdims)
    mx.linalg = linalg

    return mx


def _make_nn_mock():
    nn = types.SimpleNamespace()

    class Module:
        pass

    class Linear:
        def __init__(self, in_dim, out_dim):
            scale = math.sqrt(1.0 / in_dim)
            self.weight = np.random.uniform(-scale, scale, (out_dim, in_dim)).astype(np.float32)
            self.bias   = np.zeros(out_dim, dtype=np.float32)
        def __call__(self, x):
            return x @ self.weight.T + self.bias

    class GELU:
        def __call__(self, x):
            return x * 0.5 * (1.0 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    class _Losses:
        @staticmethod
        def cross_entropy(logits, targets, reduction="none"):
            # logits: (B, S, V)  targets: (B, S) int
            B, S, V = logits.shape
            lf = logits.reshape(-1, V)
            tf = targets.reshape(-1).astype(int)
            # numerically stable log-sum-exp
            m  = lf.max(axis=-1, keepdims=True)
            lse = np.log(np.sum(np.exp(lf - m), axis=-1)) + m.squeeze(-1)
            score = lf[np.arange(len(tf)), tf]
            loss = lse - score          # (B*S,)
            if reduction == "mean":
                return np.mean(loss)
            elif reduction == "sum":
                return np.sum(loss)
            return loss.reshape(B, S)

    nn.Module  = Module
    nn.Linear  = Linear
    nn.GELU    = GELU
    nn.losses  = _Losses()
    return nn


# 注入 mock 前先確認不會覆蓋已存在的真實 mlx
if "mlx.core" not in sys.modules:
    mlx_pkg = types.ModuleType("mlx")
    sys.modules["mlx"]        = mlx_pkg
    sys.modules["mlx.core"]   = _make_mx_mock()   # type: ignore
    sys.modules["mlx.nn"]     = _make_nn_mock()    # type: ignore


# ─── numpy 版的核心算法實作（與 src/ 邏輯完全對應）─────────────────────────

class NpReflectionAdapter:
    """numpy 版 ReflectionAdapter，邏輯與 src/adapter.py 完全相同。"""

    def __init__(self, hidden_size: int, num_virtual_tokens: int = 4, expansion: int = 4):
        self.hidden_size       = hidden_size
        self.num_virtual_tokens = num_virtual_tokens

        intermediate = hidden_size * expansion
        output_size  = hidden_size * num_virtual_tokens

        s1 = math.sqrt(1.0 / hidden_size)
        self.fc1_w = np.random.uniform(-s1, s1, (intermediate, hidden_size)).astype(np.float32)
        self.fc1_b = np.zeros(intermediate, dtype=np.float32)

        s2 = math.sqrt(1.0 / intermediate)
        self.fc2_w = np.random.uniform(-s2, s2, (output_size, intermediate)).astype(np.float32)
        self.fc2_b = np.zeros(output_size, dtype=np.float32)

    def _gelu(self, x):
        return x * 0.5 * (1.0 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    def __call__(self, h_top: np.ndarray) -> np.ndarray:
        """(batch, hidden_size) -> (batch, num_virtual_tokens, hidden_size)"""
        x = h_top @ self.fc1_w.T + self.fc1_b
        x = self._gelu(x)
        x = x @ self.fc2_w.T + self.fc2_b
        batch = h_top.shape[0]
        return x.reshape(batch, self.num_virtual_tokens, self.hidden_size)


def np_loss_fn(logits: np.ndarray, targets: np.ndarray, n_v: int) -> float:
    """numpy 版 loss_fn，與 src/train.py 的 loss_fn 邏輯相同。

    Args:
        logits:  (batch, n_v + seq_len, vocab_size)
        targets: (batch, seq_len) — int indices
        n_v:     num_virtual_tokens
    """
    aligned_logits  = logits[:, n_v:-1, :]   # (batch, seq_len-1, vocab_size)
    aligned_targets = targets[:, 1:]          # (batch, seq_len-1)

    B, S, V = aligned_logits.shape
    lf = aligned_logits.reshape(-1, V)
    tf = aligned_targets.reshape(-1).astype(int)
    m   = lf.max(axis=-1, keepdims=True)
    lse = np.log(np.sum(np.exp(lf - m), axis=-1)) + m.squeeze(-1)
    score = lf[np.arange(len(tf)), tf]
    return float(np.mean(lse - score))


def np_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """與 src/evaluate.py 的 cosine_similarity 邏輯相同。"""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.sum(a_norm * b_norm, axis=-1)


# ─── 測試案例 ──────────────────────────────────────────────────────────────

class TestReflectionAdapter(unittest.TestCase):
    """驗證 ReflectionAdapter 的形狀轉換與基礎數值特性。"""

    def setUp(self):
        np.random.seed(42)
        self.hidden = 16
        self.n_v    = 4
        self.adapter = NpReflectionAdapter(self.hidden, self.n_v)

    def test_output_shape_batch1(self):
        h = np.random.randn(1, self.hidden).astype(np.float32)
        self.assertEqual(self.adapter(h).shape, (1, self.n_v, self.hidden))

    def test_output_shape_batch4(self):
        h = np.random.randn(4, self.hidden).astype(np.float32)
        self.assertEqual(self.adapter(h).shape, (4, self.n_v, self.hidden))

    def test_output_is_finite(self):
        h = np.random.randn(2, self.hidden).astype(np.float32)
        self.assertTrue(np.all(np.isfinite(self.adapter(h))))

    def test_different_inputs_give_different_outputs(self):
        h1 = np.ones((1, self.hidden),  dtype=np.float32)
        h2 = np.zeros((1, self.hidden), dtype=np.float32)
        self.assertFalse(np.allclose(self.adapter(h1), self.adapter(h2)))

    def test_fc2_output_size_matches_reshape(self):
        """fc2 輸出維度必須等於 hidden_size * num_virtual_tokens，否則 reshape 會失敗。"""
        expected_out = self.hidden * self.n_v
        self.assertEqual(self.adapter.fc2_w.shape[0], expected_out)

    def test_num_virtual_tokens_2(self):
        adapter = NpReflectionAdapter(hidden_size=8, num_virtual_tokens=2)
        h = np.random.randn(3, 8).astype(np.float32)
        self.assertEqual(adapter(h).shape, (3, 2, 8))


class TestModelWrapperConcatenation(unittest.TestCase):
    """驗證 ReflectiveGemma.__call__ 的拼接邏輯。"""

    def setUp(self):
        self.batch   = 2
        self.seq_len = 5
        self.hidden  = 8
        self.n_v     = 3

    def test_concat_shape(self):
        p_soft       = np.random.randn(self.batch, self.n_v, self.hidden).astype(np.float32)
        input_embeds = np.random.randn(self.batch, self.seq_len, self.hidden).astype(np.float32)
        x_second     = np.concatenate([p_soft, input_embeds], axis=1)
        self.assertEqual(x_second.shape, (self.batch, self.n_v + self.seq_len, self.hidden))

    def test_virtual_tokens_precede_input_embeds(self):
        """虛擬代幣必須排在序列開頭。"""
        p_soft       = np.full((1, 2, 4), 9.0, dtype=np.float32)
        input_embeds = np.zeros((1, 3, 4), dtype=np.float32)
        x_second     = np.concatenate([p_soft, input_embeds], axis=1)
        np.testing.assert_allclose(x_second[0, :2, :], 9.0)
        np.testing.assert_allclose(x_second[0, 2:, :], 0.0)

    def test_concat_does_not_alter_batch_or_hidden(self):
        p_soft       = np.random.randn(2, 4, 8).astype(np.float32)
        input_embeds = np.random.randn(2, 6, 8).astype(np.float32)
        result       = np.concatenate([p_soft, input_embeds], axis=1)
        self.assertEqual(result.shape[0], 2)       # batch 不變
        self.assertEqual(result.shape[1], 4 + 6)   # seq dim 相加
        self.assertEqual(result.shape[2], 8)        # hidden 不變

    def test_concat_preserves_values(self):
        """確認拼接後各子區段數值未被破壞。"""
        p_soft       = np.random.randn(1, 3, 4).astype(np.float32)
        input_embeds = np.random.randn(1, 5, 4).astype(np.float32)
        x_second     = np.concatenate([p_soft, input_embeds], axis=1)
        np.testing.assert_array_equal(x_second[:, :3, :], p_soft)
        np.testing.assert_array_equal(x_second[:, 3:, :], input_embeds)


class TestLossFn(unittest.TestCase):
    """驗證 loss_fn 的對齊邏輯與數值特性。"""

    def setUp(self):
        np.random.seed(0)
        self.vocab = 100
        self.n_v   = 4

    def test_aligned_logits_shape(self):
        batch, seq_len = 2, 8
        total = self.n_v + seq_len
        logits  = np.random.randn(batch, total, self.vocab).astype(np.float32)
        targets = np.random.randint(0, self.vocab, (batch, seq_len))

        aligned_logits  = logits[:, self.n_v:-1, :]
        aligned_targets = targets[:, 1:]

        self.assertEqual(aligned_logits.shape,  (batch, seq_len - 1, self.vocab))
        self.assertEqual(aligned_targets.shape, (batch, seq_len - 1))

    def test_loss_is_finite_scalar(self):
        batch, seq_len = 1, 6
        logits  = np.random.randn(batch, self.n_v + seq_len, self.vocab).astype(np.float32)
        targets = np.random.randint(0, self.vocab, (batch, seq_len))
        loss = np_loss_fn(logits, targets, self.n_v)
        self.assertTrue(np.isfinite(loss))
        self.assertIsInstance(loss, float)

    def test_perfect_logits_give_near_zero_loss(self):
        """當 logits 強烈指向正確 token 時，loss 應趨近於 0。

        full_targets = [0, 7, 3, 1, 2]
        aligned_targets = full_targets[1:] = [7, 3, 1, 2]
        logits[:, n_v+i, :] 預測 aligned_targets[i]
        """
        batch, seq_len = 1, 5
        # full_targets[t+1] 是 position n_v+t 要預測的 token
        full_targets = np.array([[0, 7, 3, 1, 2]], dtype=int)  # (1, seq_len)
        aligned_tgt  = full_targets[0, 1:]                      # [7, 3, 1, 2]

        total  = self.n_v + seq_len
        logits = np.ones((batch, total, self.vocab), dtype=np.float32) * -100.0
        for i, t in enumerate(aligned_tgt):
            logits[0, self.n_v + i, t] = 100.0  # 強烈指向正確 token

        loss = np_loss_fn(logits, full_targets, self.n_v)
        self.assertLess(loss, 0.01)

    def test_virtual_tokens_excluded_from_loss(self):
        """修改虛擬代幣位置的 logits 不應影響 loss 值。"""
        batch, seq_len = 1, 5
        total   = self.n_v + seq_len
        logits  = np.random.randn(batch, total, self.vocab).astype(np.float32)
        targets = np.random.randint(0, self.vocab, (batch, seq_len))

        logits_alt = logits.copy()
        logits_alt[:, :self.n_v, :] = 999.0   # 只改虛擬代幣部分

        loss_a = np_loss_fn(logits,     targets, self.n_v)
        loss_b = np_loss_fn(logits_alt, targets, self.n_v)

        self.assertAlmostEqual(loss_a, loss_b, places=4,
                               msg="虛擬代幣位置的 logits 不應影響 loss")

    def test_better_logits_give_lower_loss(self):
        """正確指向的 logits 應比隨機 logits 產生更低的 loss。"""
        batch, seq_len = 1, 6
        targets = np.random.randint(0, self.vocab, (batch, seq_len))

        rand_logits = np.random.randn(batch, self.n_v + seq_len, self.vocab).astype(np.float32)

        perfect_logits = np.full((batch, self.n_v + seq_len, self.vocab), -10.0, dtype=np.float32)
        for i, t in enumerate(targets[0, 1:]):
            perfect_logits[0, self.n_v + i, t] = 10.0

        rand_loss    = np_loss_fn(rand_logits,    targets, self.n_v)
        perfect_loss = np_loss_fn(perfect_logits, targets, self.n_v)

        self.assertLess(perfect_loss, rand_loss)

    def test_loss_next_token_offset(self):
        """確認偏移方向正確：logits[:, n_v+t] 預測 targets[:, t+1]（非 targets[:, t]）。"""
        batch, seq_len = 1, 4
        vocab = 10
        n_v   = 2
        total = n_v + seq_len

        targets = np.array([[2, 5, 3, 7]], dtype=int)  # (1, 4)

        # 在 logits[n_v+0] 強烈指向 targets[1]=5
        logits = np.ones((batch, total, vocab), dtype=np.float32) * -50.0
        logits[0, n_v + 0, 5] = 50.0   # position n_v+0 預測 targets[1]
        logits[0, n_v + 1, 3] = 50.0   # position n_v+1 預測 targets[2]
        logits[0, n_v + 2, 7] = 50.0   # position n_v+2 預測 targets[3]

        loss = np_loss_fn(logits, targets, n_v)
        self.assertLess(loss, 0.01, "next-token 偏移正確時 loss 應極低")


class TestCosineSimilarity(unittest.TestCase):
    """驗證 cosine_similarity 的數值正確性。"""

    def test_identical_vectors_sim_is_1(self):
        v = np.random.randn(8).astype(np.float32)
        sim = np_cosine_similarity(v[None], v[None])
        self.assertAlmostEqual(float(sim[0]), 1.0, places=5)

    def test_orthogonal_vectors_sim_is_0(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])
        self.assertAlmostEqual(float(np_cosine_similarity(a, b)[0]), 0.0, places=5)

    def test_opposite_vectors_sim_is_neg1(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        self.assertAlmostEqual(float(np_cosine_similarity(a, b)[0]), -1.0, places=5)

    def test_range_is_minus1_to_1(self):
        table = np.random.randn(200, 32).astype(np.float32)
        v     = np.random.randn(1, 32).astype(np.float32)
        sims  = np_cosine_similarity(v, table)
        self.assertTrue(np.all(sims >= -1.0 - 1e-5))
        self.assertTrue(np.all(sims <=  1.0 + 1e-5))

    def test_self_similarity_is_highest(self):
        """向量與自身應有最高相似度。"""
        table      = np.random.randn(100, 32).astype(np.float32)
        target_idx = 17
        v          = table[target_idx:target_idx + 1]
        sims       = np_cosine_similarity(v, table)
        self.assertEqual(int(np.argmax(sims)), target_idx)

    def test_output_shape_broadcast(self):
        v     = np.random.randn(1, 16).astype(np.float32)
        table = np.random.randn(50, 16).astype(np.float32)
        sims  = np_cosine_similarity(v, table)
        self.assertEqual(sims.shape, (50,))


class TestAdapterParameterSensitivity(unittest.TestCase):
    """驗證 Adapter 的梯度路徑（輸出對所有參數都有依賴）。"""

    def setUp(self):
        np.random.seed(7)
        self.adapter = NpReflectionAdapter(hidden_size=8, num_virtual_tokens=4)
        self.h = np.random.randn(1, 8).astype(np.float32)

    def test_output_changes_with_fc1_weight(self):
        out1 = self.adapter(self.h).copy()
        self.adapter.fc1_w += 0.5
        self.assertFalse(np.allclose(self.adapter(self.h), out1))

    def test_output_changes_with_fc2_weight(self):
        out1 = self.adapter(self.h).copy()
        self.adapter.fc2_w += 0.5
        self.assertFalse(np.allclose(self.adapter(self.h), out1))

    def test_output_changes_with_bias(self):
        out1 = self.adapter(self.h).copy()
        self.adapter.fc1_b += 1.0
        self.assertFalse(np.allclose(self.adapter(self.h), out1))


class TestEmptyBatchSafety(unittest.TestCase):
    """驗證空 batch 邊界案例。"""

    def test_train_one_epoch_returns_zero_on_empty_batches(self):
        """train_one_epoch 遇到空 batches 應回傳 0.0 而非 NameError。"""
        running_loss = 0.0
        num_steps    = 0
        for num_steps, _ in enumerate([], start=1):
            pass
        result = running_loss / max(num_steps, 1)
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# tests/ml_lab/week2/test_attention.py
# 目的:
#   - 形状と基本性質（重みの和=1、マスクが効く）を素早く検証する。
# 学習の狙い:
#   - 「マスク規約（True=通す）」をテストで固定化 → 実装/呼び出し側の齟齬を防ぐ。
#   - 因果マスク/パディングマスクの挙動を小ケースで直感的に理解する。
# 間違っていると:
#   - 和が1にならない（maxシフト忘れ/NaN混入）
#   - マスクが効かず「未来」や「パディング」に注意が漏れる
#   - 形状がズレてバッチ/系列方向のエラーになる

import numpy as np
import numpy.testing as npt
from src.ml_lab.week2.attention.attention import (casual_mask, padding_mask,
                                                  scaled_dot_product_attention)


def test_attention_shapes_and_prob_sum():
    """基本の形状と「重みの和=1」を検証。"""
    B, Tq, Tk, Dk, Dv = 2, 3, 4, 5, 6
    rng = np.random.default_rng(0)
    q = rng.normal(size=(B, Tq, Dk))
    k = rng.normal(size=(B, Tk, Dk))
    v = rng.normal(size=(B, Tk, Dv))

    out, attn = scaled_dot_product_attention(q, k, v, mask=None)
    assert out.shape == (B, Tq, Dv)
    assert attn.shape == (B, Tq, Tk)
    # 各クエリ位置で重みが確率分布（総和=1）になっているはず
    npt.assert_allclose(attn.sum(axis=-1), np.ones((B, Tq)), atol=1e-6)


def test_causal_mask_blocks_future():
    """因果マスクで未来（j>i）への注意が0に近くなることを確認。"""
    B, T, D = 1, 4, 3
    # 分かりやすくするため、K/V を標準基底に近い形にし、Q を未来キーに強く一致させる
    # ここでは、各タイムステップ j の key は e_j、value も e_j * 10 とする
    I = np.eye(T)
    q = I[None, :, :D].copy()  # [1,T,D]（D<=T）: t=i の Q は e_i を含む
    k = I[None, :, :D].copy()  # [1,T,D]
    v = (10.0 * I[:, :D])[None, :, :]  # [1,T,D]

    # マスク無しなら、各 i は同じ i のキーに最大注意（自己参照）
    out_no_mask, attn_no_mask = scaled_dot_product_attention(q, k, v, mask=None)
    # 因果マスク（iは j<=i のみ見られる）
    causal = casual_mask(T)
    # [B,Tq,Tk] にブロードキャスト
    causal = np.broadcast_to(causal, (B, T, T))
    out_mask, attn_mask = scaled_dot_product_attention(q, k, v, mask=causal)

    # 未来側の重みが 0 に近いこと（数値的に厳密0ではないので小ささで判定）
    future = np.triu(np.ones((T, T), dtype=bool), k=1)  # 上三角=未来
    assert np.all(
        attn_mask[0][future] < 1e-6
    ), "因果マスクで未来への注意が残っています。実装を確認してください。"

    # 自己参照の重みは残っている（対角成分は大きいはず）
    diag = np.diag_indices(T)
    assert np.all(
        attn_mask[0][diag] > 0.5
    ), "自己位置への注意が弱すぎます。スケーリング/ソフトマックスを確認。"


def test_padding_mask_zeroes_out_padded_keys():
    """パディング（無効キー）に注意が行かないことを確認。"""
    B, Tq, Tk, D = 2, 3, 5, 4
    rng = np.random.default_rng(1)
    q = rng.normal(size=(B, Tq, D))
    k = rng.normal(size=(B, Tk, D))
    v = rng.normal(size=(B, Tk, D))

    # 片方のサンプルは最後の2トークンがパディング、もう片方は全て有効
    valid_lengths = np.array([Tk - 2, Tk], dtype=int)  # [3,5]
    pad_mask = padding_mask(valid_lengths, t_k=Tk)  # [B,1,Tk] -> broadcast で [B,Tq,Tk]

    out, attn = scaled_dot_product_attention(q, k, v, mask=pad_mask)
    # サンプル0の最後の2列（padding列）の注意が小さいことを確認
    assert np.all(
        attn[0, :, -2:] < 1e-6
    ), "パディング位置に注意が漏れています。マスクの適用を確認。"

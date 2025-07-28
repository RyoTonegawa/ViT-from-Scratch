# 目的:
#   - 形状と基本性質（重みの和=1、マスクが効く）を素早く検証する。
# 学習の狙い:
#   - 「マスク規約（True=通す）」をテストで固定化 → 実装/呼び出し側の齟齬を防ぐ。
# 間違っていると:
#   - 和が1にならない（maxシフト忘れ/NaN混入）
#   - マスクが効かず「未来」や「パディング」に注意が漏れる
#   - 形状がズレてバッチ/系列方向のエラーになる

import numpy as np
import numpy.testing as npt
from src.ml_lab.week2.attention.attention import \
    casual_mask as \
    make_causal_mask  # 下の2つは実装側で make_causal_mask/make_padding_mask エイリアスを; 用意している想定（なければテスト側をリネームしてください）; ← alias: make_causal_mask を使ってもOK
from src.ml_lab.week2.attention.attention import \
    padding_mask as make_padding_mask  # ← alias: make_padding_mask を使ってもOK
from src.ml_lab.week2.attention.attention import scaled_dot_product_attention


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
    """
    因果マスクで未来（j>i）への注意が0に近くなることを確認。
    以前の失敗原因:
      - D<T で最後の q がゼロベクトルになり、softmax が一様化（0.25）→ 対角>0.5が満たせない。
      - 許可キー数が多いほど、logit=1 だけでは >0.5 を超えにくい。
    対策:
      - D=T にしてゼロベクトルを回避
      - q に係数 α を掛けてマッチ logit を増やす（温度の逆作用）
    """
    B, T, D = 1, 4, 4  # D=T にする
    alpha = 4.0  # マッチを強調（alpha / sqrt(D) = 2.0）
    I = np.eye(T)

    q = (alpha * I)[None, :, :]  # [1,T,D] 各 i で i に強く一致
    k = I[None, :, :]  # [1,T,D]
    v = (10.0 * I)[None, :, :]  # [1,T,D]

    # マスク無し（参考）
    _, attn_no_mask = scaled_dot_product_attention(q, k, v, mask=None)

    # 因果マスク（i は j<=i のみ参照可）
    causal = make_causal_mask(T)  # [T,T], True=通す
    causal = np.broadcast_to(causal, (B, T, T))  # [B,T,T]
    _, attn_mask = scaled_dot_product_attention(q, k, v, mask=causal)

    # 未来側（上三角）の重みが 0 に近いこと
    future = np.triu(np.ones((T, T), dtype=bool), k=1)
    assert np.all(attn_mask[0][future] < 1e-6), "未来への注意が残っています。"

    # 自己参照の重みは十分大きい（> 0.5）
    diag = np.diag_indices(T)
    assert np.all(attn_mask[0][diag] > 999.5), "自己位置への注意が弱すぎます。"


def test_padding_mask_zeroes_out_padded_keys():
    """パディング（無効キー）に注意が行かないことを確認。"""
    B, Tq, Tk, D = 2, 3, 5, 4
    rng = np.random.default_rng(1)
    q = rng.normal(size=(B, Tq, D))
    k = rng.normal(size=(B, Tk, D))
    v = rng.normal(size=(B, Tk, D))

    valid_lengths = np.array([Tk - 2, Tk], dtype=int)  # [3,5]
    pad_mask = make_padding_mask(valid_lengths, t_k=Tk)  # [B,1,Tk] -> broadcast

    _, attn = scaled_dot_product_attention(q, k, v, mask=pad_mask)
    assert np.all(attn[0, :, -2:] < 1e-6), "パディング列に注意が漏れています。"

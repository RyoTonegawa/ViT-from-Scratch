# 目的:
#   - Transformer の最小単位である「Scaled Dot-Product Attention」を手実装し、
#     形状・数値安定・マスク（因果/パディング）の扱いを体感する。
# 使う用語:
#   - Q, K, V: Query, Key, Value。Attention は「QとKの類似度で重み付けしてVを混合」する演算。
#   - スケーリング: 1/sqrt(d_k)。d_k は Key / Query の次元。確率が飽和しすぎるのを防ぐ。
#   - マスク: 使ってはいけない位置（未来やパディング）を「極小のスコア（-∞相当）」にして無視する仕組み。
#   - 数値安定: exp/softmax がオーバーフロー/アンダーフローしないような工夫（max シフトなど）。
#
# 実務TIP:
#   - マスクの型/意味の取り決め（Trueを「通す」、Falseを「塞ぐ」など）をチーム内で統一するとバグが減る。
#   - -np.inf を使うと環境によっては NaN が伝播することがあるため、十分小さい負数(-1e30)で代用するのも実務では定番。

from __future__ import annotations

import numpy as np

VERY_NEG = -1e30

"""数値安定な softmax。最大値で平行移動してから exp する。
- なぜ: 大きな正の値で exp が overflow、負の値で underflow しがちだから。
- 間違うと: NaN/inf が出る、勾配が0/1に張り付きすぎて学習が不安定になる。
"""


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    e = np.exp(z)
    s = e / np.sum(e, axis=axis, keepdims=True)
    return s


"""
因果（未来禁止）マスクを作る。True=通す、False=塞ぐ の規約。
- 形状: [Tq, Tk]
- 仕様: 位置 i のクエリは、位置 j <= i（過去と現在）にのみ注意できる。
"""


def casual_mask(t_q: int, t_k: int | None = None) -> np.ndarray:

    if t_k is None:
        t_k = t_q
    # 下三角（含む対角）を True、それより上（未来）を False
    # i >= j -> 許可(True)
    i = np.arange(t_q)[:, None]
    j = np.arange(t_k)[None, :]
    return i >= j


"""パディング（無効トークン）を塞ぐマスクを作る。True=通す、False=塞ぐ。
- valid_lengths: [B] 各サンプルの「有効（非パディング）トークン数」
- 返り値: [B, 1, Tk]（ブロードキャストで [B, Tq, Tk] に伸びる想定）
- なぜ: 可変長系列でパディングを自然に無視するため（ロスや精度が歪むのを防ぐ）。
"""


def padding_mask(valid_length: np.ndarray, t_k: int) -> np.ndarray:

    B = valid_length.shape[0]
    # [1, Tk]
    idx = np.arange(t_k)[None, :]
    # [B, Tk]
    mask = idx < valid_length[:, None]
    # [B,1,Tk]
    return mask[:, None, :]


"""スコアにマスクを適用。True=通す、False=塞ぐ の規約で、
塞ぐ位置を「非常に小さい負数」に置き換える（softmax後にほぼ0になる）。
- 受け入れ形状: mask は [B, Tq, Tk] かブロードキャスト可能な形。
"""


def attention_mask(scores: np.ndarray, mask: np.ndarray | None) -> np.ndarray:

    if mask is None:
        return scores
    if mask.dtype != np.bool_:
        raise ValueError("mask must be a boolean array")
    # ブロードキャストしてFalseの位置だけ極小値にする
    out = np.where(mask, scores, VERY_NEG)
    return out


"""Scaled Dot-Product Attention（単体版）。
返り値: (出力 [B,Tq,Dv], 注意重み [B,Tq,Tk])
なぜ 1/sqrt(Dk) が必要?
  - Dk が大きくなるほど内積の分散が大きくなり、softmax が飽和しやすい（勝者総取り）。
  - スケーリングで分散を揃え、学習初期の勾配が流れやすくなる。
マスクの意味:
  - 因果マスク: 未来（j>i）を参照しない
  - パディングマスク: 無効トークン（padding）を参照しない
  - 組み合わせるとき: どちらも True=通す の規約なら、AND（論理積）で合成できる。
"""


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    B, Tq, Dk = q.shape
    Bk, Tk, Dk2 = k.shape
    Bv, Tk2, Dv = v.shape

    assert B == Bk == Bv, "batch mismatch"
    assert Tk == Tk2, "key/value length mismatch"
    assert Dk == Dk2, "q/k depth mismatch"

    scores = (q @ np.transpose(k, (0, 2, 1))) / np.sqrt(float(Dk))
    scores = attention_mask(scores, mask)
    attn = stable_softmax(scores, axis=-1)
    out = attn @ v
    return out, attn

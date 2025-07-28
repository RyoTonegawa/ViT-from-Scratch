"""
# 目的:
#   - 物体検知/CLIP/SigLIPの前段として基礎となる活性化関数を、
#     「なぜその式になるのか」「数値安定とは何か」を実装とコメントで理解する。
# ポリシー:
#   - 依存は NumPy のみ（学習のために低レベルに書く）
#   - 実務でトラブルになりがちな数値不安定箇所を、安定化版で実装
#
# 用語解説:
#   - ロジット(logit): シグモイドやソフトマックスに入る "生のスコア"（未正規化の値）
#   - 数値安定(numerical stability): 端の値でオーバーフロー/アンダーフロー/NaNにならない工夫
#   - log-sum-expトリック: log(sum(exp(x))) を安定に計算するための「最大値での平行移動」

"""

from __future__ import annotations

from typing import Final

import numpy as np

LOG_MAX_FLOAT32: Final[float] = float(np.log(np.finfo(np.float32).max))  # ≈ 88.72
NEGATIVE_BOUND = -LOG_MAX_FLOAT32

"""Sigmoid。
アイデア: 非常に負の値で exp がアンダーフローし、非常に正の値でオーバーフローしやすい。
        入力をクリップしてから 1/(1+exp(-x)) を計算する。
何を意図しているか:
  - 端の値でも NaN/inf にならず、常に (0,1) の確率っぽい値が出るようにする。
実務TIP:
  - BCE(二値交差エントロピー)を自作する時は「with logits」版の安定式を用いる（後の課題）。
  

clipは配列の各要素を区間[a_min,a_max]にはさむ関数。
ここでは最大88,最小−88に制限してからexpを計算し、オーバーフロー、アンダーフローを抑える
    →float３２の実用上の閾値に由来する。
    clipをすると閾値で微分不可能になり、閾値の上下が勾配０になる、”ハードな飽和”になる。
    
注意：クリップは境界点で勾配が途切れる（非微分）ため、通常は境界が学習域から十分遠い値に設定します。
"""


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), NEGATIVE_BOUND, LOG_MAX_FLOAT32)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_unstable(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -88, 88)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    # (1) 出力配列を用意。dtypeは「xとfloat32の上位互換」にする
    #     - xがfloat16/整数なら少なくともfloat32に昇格
    #     - xがfloat64ならfloat64のまま（精度を落とさない）
    out = np.empty_like(x, dtype=np.result_type(x, np.float32))

    # (2) 符号で領域分割：x>=0 と x<0
    pos = x >= 0

    # --- x>=0 側（安全に exp(-x) を計算できる：値域は [0,1]）---
    # (3) ここでは exp(-x) が小さく、オーバーフローしない
    z = np.exp(-x[pos])

    # (4) 通常の式で OK
    out[pos] = 1.0 / (1.0 + z)

    # --- x<0 側（exp(-x) は巨大になり得るので避ける）---
    # (5) 代わりに exp(x) を計算（x<0 なので値域は [0,1] で安全）
    z = np.exp(x[pos])

    # (6) 同値変形した式：sigma(x) = e^x / (1 + e^x)
    #     分母分子を e^x で割ることで、巨大な exp(-x) を回避
    out[~pos] = z / (1.0 + z)

    return out


"""数値安定なソフトマックス（温度 τ つき）。
安定化: max シフト (x - max(x)) をしてから exp → 正規化

temperature(温度)とは:
  - τ を小さくすると分布がシャープ（勝者がより勝つ）になり、大きくするとフラットになる。
  - CLIPの InfoNCE 損失で重要なハイパラ。微調整で検索性能やゼロショット性能に影響。

意図:
  - バッチ/クラス方向で確率分布（総和=1）を作る。極端なロジットでも NaNにしない。
"""


def softmax(x: np.ndarray, axis: int = -1, tempreture: float = 1.0) -> np.ndarray:
    if tempreture <= 0:
        raise ValueError("tempreture must be > 0")
    z = x.astype(np.float64) / float(tempreture)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    s = e / np.sum(e, axis=axis, keepdims=True)
    return s


"""log(sum(exp(x))) を安定に計算。
トリック: m = max(x); log(sum(exp(x))) = m + log(sum(exp(x - m)))
何に使う?
- 交差エントロピーの安定計算や、確率の正規化(対数空間)で頻出。
- ソフトマックスの内部で自然に現れる量。
"""


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))


# def label_somoothing_targets(
#     num_classes: int,
#     target_indices:np.ndarray,
#     smoothing:float,
# )-> np.ndarray:

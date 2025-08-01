# **実務ポイント（物体検知・CLIP/SigLIP視点）**
- IoU は検出の基礎。IoU閾値（例0.5 / COCOは0.5〜0.95平均）がTP/FPを分け、mAPに直結。
- CLIP はsoftmax(InfoNCE)で“相対ランク”を強く学習。SigLIP はsigmoid(BCE)で“ペア独立”に学習し多正解に強い。
- 確率のキャリブレーション（ECE）は「確率の当たり具合」。閾値最適化や温度スケーリングで現場精度が上がる。


# **補足用語解説**

- logits: 活性化前の連続スコア（例：線形層の出力）。確率ではない。
- InfoNCE: softmax + CE によるコントラスト損失。バッチ内で正例と負例を相対比較。
- 温度 τ: softmax の鋭さを調整するスカラー。小さいと勝者総取りに近づき、大きいと均す。
- BCE（with logits）: sigmoid + BCE を安定に計算する実装。マルチラベルや多正解に向く。
- ECE: 予測確率の「当たり具合」。信頼度と実際の正答率のズレを測る。
- しきい値最適化: 連続スコア→二値判定の境界を検証データで最適化（F1最大など）。
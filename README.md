# ViT-from-Scratch
this repository contains ViT implementations

# ToDo　ML

1. 評価指標
2. アテンション実装（単体→Multi-Head→ブロック）
3. ミニGPT（デコーダオンリー）
4. エンコーダ—デコーダ＋クロスアテンション
5. ViT-Tiny（CIFAR-10）
6. CLIP損失（InfoNCE/BCE）
7. LiT（画像塔凍結＋テキスト塔学習）

# ToDo math

1. **線形代数（必修）**
- **到達目標**
 ベクトル/行列、ノルム、直交射影、固有分解、SVD/擬似逆行列、トレース/フロベニウスノルム、コサイン類似度。
- **実装課題**
位置埋め込み（sin/cos）の公式を自作 → 相関を可視化。
重み初期化（Xavier/He）を実装して収束の差を観察。

2. **微積分 & 行列微分**
- **到達目標**: 連鎖律、ヤコビアン/ヘッセ行列、softmax/LayerNorm の勾配、クロスエントロピーの導出。
- **実装課題**
softmax+CE の手導出と PyTorch の autograd.gradcheck で検証。
Scaled Dot-Product Attention のスケーリング（1/√d）の意味を数式で確認。

3. **確率・統計 & 情報理論**
- **到達目標**
期待値・分散、KL/JS、最大尤度、交差エントロピーとInfoNCE、温度スケーリング、ベイズの直観。
- **実装課題**
InfoNCE（softmax）と BCE（sigmoid）を自作し、温度 τ のスイープで挙動比較。

4. **最適化**
- **到達目標**　
SGD/モメンタム/AdamW、学習率スケジュール（cosine、warmup）、重み減衰とL2の違い、勾配爆発・消失対策（clip、残差、LayerNorm）。
- **実装課題**
1バッチ overfit 実験で実装の健全性確認（損失が0近くまで下がるか）。

5. **変換器（Transformer）の数理**
- **到達目標**　
自己注意・クロス注意（Qはデコーダ、K/Vはエンコーダ）、複雑度 O(L²d)、マスクの意味、Multi-Head がもたらす表現分解、位置埋め込みの役割。
- **実装課題**
Encoder/Decoderブロックを素実装、トイ翻訳で BLEU を計測。

6. **表現学習の損失設計**
- **到達目標**　
コントラスト学習（InfoNCE/BCE）、センタリング/温度/EMA（DINO, BYOL系）、**プロトタイプ割当（SwAV）**の直観。
- **実装課題**
ミニCLIP（画像塔＋テキスト塔）→ CIFAR-10 ゼロショット。
LiT：画像塔を凍結し、テキスト塔のみ学習。

## パッケージ追加
```bash
uv add package-name
```

## 環境を最初に有効化
```bash
source .venv/bin/activate
```

## コードの実行
```bash
python -m ml_lab.check_mps
```

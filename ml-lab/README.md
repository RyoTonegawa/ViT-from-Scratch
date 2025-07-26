## 環境設定

```plain text
brew install pyenv pipx direnv
# 推奨: シェルに pyenv 初期化を追記（~/.zshrc など）
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

pyenv install 3.12.4   # 例
pyenv global 3.12.4    # システム標準は触らない

pipx install uv

# 新規プロジェクト（例：ml-lab）
mkdir ml-lab && cd ml-lab
uv init --package  # pyproject.toml を生成
uv venv            # .venv を作成
source .venv/bin/activate

# 科学計算・DL 基本
uv add torch torchvision torchaudio
uv add numpy scipy matplotlib pandas scikit-learn
# 実験用
uv add jupyterlab ipykernel
# 実装補助（任意）
uv add transformers datasets accelerate
# 品質（Linter/Formatter）
uv add --dev ruff black isort pre-commit pytest


MPS フォールバックを許す場合（未対応演算をCPUに逃がす）：
echo 'export PYTORCH_ENABLE_MPS_FALLBACK=1' >> .envrc
direnv allow   # カレントに入ると自動で環境変数を読み込む運用も便利

# check_mps.py
import torch
print("MPS available:", torch.backends.mps.is_available())
device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.randn(2048, 2048, device=device)
y = x @ x.t()
print(device, y.mean().item())

python check_mps.py

```

# リポジトリについて

このリポジトリは、OpenAIのGPT-2モデルをスクラッチ開発するためのものです

## ライセンス

このプロジェクトは 下記の引用よりライセンスを継承しています

- **LLMs-from-scratch** by rasbt  
  Apache License 2.0 ライセンスを継承しています  
  https://www.apache.org/licenses/  
  元のリポジトリ  
  https://github.com/rasbt/LLMs-from-scratch  

- **openai-community/gpt2** by Hugging Face  
  MIT License ライセンスを継承しています  
  https://choosealicense.com/licenses/mit/  
  モデルページ  
  https://huggingface.co/openai-community/gpt2  


## リポジトリのクローン

```bash
git clone https://github.com/ArekuNoimar/gpt-2.git
```

## 動作要件

Python  : Python 3.10.12  
OS  : Ubuntu 22.04.5 Desktop LTS  
CUDA  : 12.4  
GPU : Nvidia Geforce RTX 4090 Laptop  

## 構成について

```txt
gpt-2/
├── Dockerfile                        // DockerImageを作成用
├── README.md                         // 解説用
├── requirements.txt                  // 依存ライブラリのインストール用
└── src
    ├── gpt2_demo.py                  // 小規模なモデルの構築&デモ
    ├── gpt2_from_scratch.py          // GPT-2の学習用
    ├── inference.py                  // GPT-2の推論用
```

### gpt2_demo.py

<ファイルの種類>
  - GPT-2実装の各機能デモンストレーション

<機能>
  - モデルアーキテクチャの詳細表示
  - 異なるパラメータでのテキスト生成デモ
  - 小規模な訓練プロセスのデモ
  - アテンション可視化の概念説明

<実行コマンド>

```bash
python3 gpt2_demo.py
```

<出力>
  - モデル仕様詳細（50,257語彙、384次元、6ヘッド等）
  - 複数設定での生成例比較
  - 訓練進行状況（2エポック）
  - 実装済み機能チェックリスト


### gpt2_from_scratch.py

<ファイルの種類>
  - GPT-2モデルの核となる実装ファイル

<機能>
  - GPT-2アーキテクチャの完全な実装（MultiHeadAttention、TransformerBlock、GPT2Model）
  - トレーニング用のTextDatasetクラスとtrain_gpt2関数
  - モデルの保存・読み込み機能
  - テキスト生成機能（temperature、top-kサンプリング対応）

<実行コマンド>

```bash
# HuggingFace 'openai-community/gpt2'のパラメーターに該当
# 使用GPU容量：3,908MiB

python3 gpt2_from_scratch.py \
    --d_model 768 \
    --n_heads 12 \
    --n_layers 12 \
    --d_ff 3072 \
    --max_len 1024 \
    --dropout 0.1 \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --num_epochs 10 \
    --save_dir ./checkpoints \
    --prompt "The quick brown"
```

```bash
# HuggingFace 'openai-community/gpt2-medium'のパラメーターに該当
# 使用GPU容量：8,576MiB

python3 gpt2_from_scratch.py \
    --d_model 1024 \
    --n_heads 16 \
    --n_layers 24 \
    --d_ff 4096 \
    --max_len 1024 \
    --dropout 0.1 \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --num_epochs 10 \
    --save_dir ./checkpoints \
    --prompt "The quick brown"
```

```bash
# HuggingFace 'openai-community/gpt2-large'のパラメーターに該当
# 使用GPU容量：17,614MiB

python3 gpt2_from_scratch.py \
    --d_model 1280 \
    --n_heads 20 \
    --n_layers 36 \
    --d_ff 5120 \
    --max_len 1024 \
    --dropout 0.1 \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --num_epochs 10 \
    --save_dir ./checkpoints \
    --prompt "The quick brown"
```

<出力>

  - デバイス情報（CUDA/CPU）
  - モデルパラメータ数
  - 訓練進行状況（エポック、バッチ、ロス値）
  - チェックポイント保存メッセージ
  - テキスト生成例（"The quick brown"から始まる生成文）
  - 最終モデル保存先パス


### inference.py 

<ファイルの種類>
  - 訓練済みモデルでの推論専用スクリプト

<機能>
  - 保存されたチェックポイントからモデル読み込み
  - インタラクティブなテキスト生成モード
  - コマンドライン引数での単発推論
  - 生成パラメータの動的変更機能

<実行コマンド>

```bash
python3 inference.py \
    --model 'checkpoints/gpt2_final_model.pt' \
    --prompt 'hello !' \
    --max_tokens 50 --temperature 0.7 \
    --top_k 40 \
    --device 'auto'
```

<出力>
  - モデル設定情報（語彙サイズ、次元数、レイヤー数など）
  - プロンプトと生成されたテキスト
  - インタラクティブモードでは対話的な生成セッション
  - 生成時間とパフォーマンス統計


## ローカル環境を利用する場合(ubuntu)

- 仮想環境の作成と有効化

```bash
python3 -m venv .venv

source .venv/bin/activate
```

- ライブラリのインストール

```bash
pip install -r requirements.txt
``` 

## Dockerを利用する場合(ubuntu)

### Dockerセットアップ

[Docker](https://docs.docker.com/engine/install/ubuntu/)より参照

- Dockerのaptリポジトリを設定

```bash
# Add Docker's official GPG key:
sudo apt-get update

sudo apt-get install ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings

sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc

sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
```

- パッケージのインストール

```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

- 権限変更

```bash
sudo groupadd docker

sudo usermod -aG docker $USER
```

### Nvidia-Container-Toolkitを設定

[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

- Nvidia Container Toolkitのaptリポジトリを設定

```bash
# Add Nvidia-Container-Toolkit official GPG key:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
```

- パッケージのインストール

```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

- サービス設定

```bash
# Add daemon.json
sudo touch  /etc/docker/daemon.json

sudo vim /etc/dcker/daemon.json
```
```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
```

### DockerImageの構築

```bash
docker build -t gpt2-dev .

docker run -it --gpus all gpt2-dev /bin/bash
```

### モデルの学習サンプル

例：'openai-community/gpt2-large'級モデルの場合

```bash
# HuggingFace 'openai-community/gpt2-large'のパラメーターに該当
# 使用GPU容量：17,614MiB

python3 gpt2_from_scratch.py \
    --d_model 1280 \
    --n_heads 20 \
    --n_layers 36 \
    --d_ff 5120 \
    --max_len 1024 \
    --dropout 0.1 \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --num_epochs 10 \
    --save_dir ./checkpoints \
    --prompt "The quick brown"
```

モデルパラメーター： 838.2M

```txt
GPT2Model(
  (token_embedding): Embedding(50257, 1280)
  (position_embedding): Embedding(1024, 1280)
  (transformer_blocks): ModuleList(
    (0-35): 36 x TransformerBlock(
      (attn): MultiHeadAttention(
        (w_q): Linear(in_features=1280, out_features=1280, bias=False)
        (w_k): Linear(in_features=1280, out_features=1280, bias=False)
        (w_v): Linear(in_features=1280, out_features=1280, bias=False)
        (w_o): Linear(in_features=1280, out_features=1280, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (mlp): MLP(
        (c_fc): Linear(in_features=1280, out_features=5120, bias=True)
        (c_proj): Linear(in_features=5120, out_features=1280, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (layer_norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      (layer_norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (layer_norm): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
  (output_projection): Linear(in_features=1280, out_features=50257, bias=False)
  (dropout): Dropout(p=0.1, inplace=False)
)
```

```log
Starting training...
Checkpoints will be saved to: ./checkpoints
Epoch 1/10, Batch 0, Loss: 11.1640
Epoch 1/10 completed. Average Loss: 6.2374
Model saved to: ./checkpoints/gpt2_checkpoint_epoch_1_20250804_043211.pt
Checkpoint saved: ./checkpoints/gpt2_checkpoint_epoch_1_20250804_043211.pt
Epoch 2/10, Batch 0, Loss: 4.5080
Epoch 2/10 completed. Average Loss: 4.0992
Model saved to: ./checkpoints/gpt2_checkpoint_epoch_2_20250804_043223.pt
Checkpoint saved: ./checkpoints/gpt2_checkpoint_epoch_2_20250804_043223.pt
Epoch 3/10, Batch 0, Loss: 3.8718
Epoch 3/10 completed. Average Loss: 3.6079
Model saved to: ./checkpoints/gpt2_checkpoint_epoch_3_20250804_043236.pt
Checkpoint saved: ./checkpoints/gpt2_checkpoint_epoch_3_20250804_043236.pt
Epoch 4/10, Batch 0, Loss: 3.3078
Epoch 4/10 completed. Average Loss: 3.1260
Model saved to: ./checkpoints/gpt2_checkpoint_epoch_4_20250804_043248.pt
Checkpoint saved: ./checkpoints/gpt2_checkpoint_epoch_4_20250804_043248.pt
Epoch 5/10, Batch 0, Loss: 2.7470
Epoch 5/10 completed. Average Loss: 2.3654
Model saved to: ./checkpoints/gpt2_checkpoint_epoch_5_20250804_043300.pt
Checkpoint saved: ./checkpoints/gpt2_checkpoint_epoch_5_20250804_043300.pt
Epoch 6/10, Batch 0, Loss: 1.7899
Epoch 6/10 completed. Average Loss: 1.4322
Model saved to: ./checkpoints/gpt2_checkpoint_epoch_6_20250804_043313.pt
Checkpoint saved: ./checkpoints/gpt2_checkpoint_epoch_6_20250804_043313.pt
Epoch 7/10, Batch 0, Loss: 0.9120
Epoch 7/10 completed. Average Loss: 0.7984
Model saved to: ./checkpoints/gpt2_checkpoint_epoch_7_20250804_043325.pt
Checkpoint saved: ./checkpoints/gpt2_checkpoint_epoch_7_20250804_043325.pt
Epoch 8/10, Batch 0, Loss: 0.5499
Epoch 8/10 completed. Average Loss: 0.4955
Model saved to: ./checkpoints/gpt2_checkpoint_epoch_8_20250804_043338.pt
Checkpoint saved: ./checkpoints/gpt2_checkpoint_epoch_8_20250804_043338.pt
Epoch 9/10, Batch 0, Loss: 0.3162
Epoch 9/10 completed. Average Loss: 0.3784
Model saved to: ./checkpoints/gpt2_checkpoint_epoch_9_20250804_043350.pt
Checkpoint saved: ./checkpoints/gpt2_checkpoint_epoch_9_20250804_043350.pt
Epoch 10/10, Batch 0, Loss: 0.2712
Epoch 10/10 completed. Average Loss: 0.2690
Model saved to: ./checkpoints/gpt2_checkpoint_epoch_10_20250804_043403.pt
Checkpoint saved: ./checkpoints/gpt2_checkpoint_epoch_10_20250804_043403.pt
Model saved to: ./checkpoints/gpt2_final_model.pt
Final model saved: ./checkpoints/gpt2_final_model.pt

Testing text generation...
Prompt: The quick brown
Generated: The quick brownPT-2 model.
     Machine learning is fascinating. Deep learning models like transformers have revolutionized natural language processing.
```

### 構築したモデルの推論サンプル

```bash
# 使用GPU容量：13,704MiB 

python3 inference.py \
    --model 'checkpoints/gpt2_final_model.pt' \
    --prompt 'hello !' \
    --max_tokens 50 --temperature 0.7 \
    --top_k 40 \
    --device 'auto'
```

```log
Model loaded successfully!
Model configuration:
  - Vocabulary size: 50257
  - Model dimension: 1280
  - Number of layers: 36
  - Max sequence length: 1024

Prompt: hello !
Generating...

Generated text:
hello !.
        The quick brown fox jumps over the lazy dog. This is a sample text for training our GPT-2 model. Large language models like transformers have revolutionized natural language processing.
```
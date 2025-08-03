# ベースイメージ
FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# システムパッケージの更新とツール類のインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      git \ 
      iputils-ping \
      net-tools \
      curl \            
      wget \                 
      ca-certificates \      
      build-essential \      
      vim \             
      less \     
      net-tools \          
      dnsutils \
&& rm -rf /var/lib/apt/lists/*

# pip のアップグレード
RUN python3 -m pip install --upgrade pip

# 必要ライブラリのインストール
RUN python3 -m pip install \
      "torch>=2.3.0" \
      "tiktoken>=0.5.1" \
      "matplotlib>=3.7.1" \
      "tensorflow>=2.18.0" \
      "tqdm>=4.66.1" \
      "numpy>=1.26,<2.1" \
      "pandas>=2.2.1" \
      "psutil>=5.9.5"

# 作業ディレクトリ
WORKDIR /src

COPY src/ /src/




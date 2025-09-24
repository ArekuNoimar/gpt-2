"""
GPT-2モデルのスクラッチ実装
"Attention Is All You Need"論文とOpenAIのGPT-2論文で説明されたアーキテクチャに基づく
"""

"""
uv run src/test_gpt2_from_scratch.py \
    --d_model 768 \
    --n_heads 12 \
    --n_layers 12 \
    --d_ff 3072 \
    --max_len 1024 \
    --dropout 0.01 \
    --batch_size 10 \
    --learning_rate 0.0001 \
    --num_epochs 1 \
    --save_dir ./checkpoints \
    --save_every 10000 \
    --sequence_length 768 \
    --prompt "what is machne learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import tiktoken
import json
import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import argparse
from datasets import load_dataset
from tqdm import tqdm


class MultiHeadAttention(nn.Module):
    """マルチヘッドアテンション機構。
    
    "Attention Is All You Need"論文で説明された、複数のヘッドを持つ
    スケールドドット積アテンションを実装する。
    
    Args:
        d_model (int): モデルの次元数。
        n_heads (int): アテンションヘッドの数。
        dropout (float): ドロップアウト確率。
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """スケールドドット積アテンションを計算する。
        
        Args:
            Q (torch.Tensor): クエリテンソル (batch_size, n_heads, seq_len, d_k)
            K (torch.Tensor): キーテンソル (batch_size, n_heads, seq_len, d_k)
            V (torch.Tensor): バリューテンソル (batch_size, n_heads, seq_len, d_k)
            mask (torch.Tensor, optional): アテンションマスク。Noneの場合は使用しない。
            
        Returns:
            tuple: アテンション出力 (batch_size, n_heads, seq_len, d_k) とアテンション重み。
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V), attention_weights
    
    def forward(self, x, mask=None):
        """マルチヘッドアテンションの順伝播を実行する。
        
        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): アテンションマスク。Noneの場合は使用しない。
            
        Returns:
            torch.Tensor: アテンション出力 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # 線形変換とヘッドへの分割
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # アテンションの適用
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # ヘッドの連結
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 最終線形射影
        output = self.w_o(attn_output)
        
        return output


class MLP(nn.Module):
    """MLP（多層パーセプトロン）- HuggingFace互換のフィードフォワードネットワーク。
    
    GELU活性化関数とドロップアウトを持つ2層フィードフォワードネットワーク。
    
    Args:
        d_model (int): モデルの次元数。
        d_ff (int): フィードフォワード層の次元数。
        dropout (float): ドロップアウト確率。
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """MLPの順伝播を実行する。
        
        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: MLP出力 (batch_size, seq_len, d_model)
        """
        return self.c_proj(self.dropout(F.gelu(self.c_fc(x))))


class PositionalEncoding(nn.Module):
    """正弦波関数を使用した位置エンコーディング。
    
    Args:
        d_model (int): モデルの次元数。
        max_len (int): 最大シーケンス長。デフォルトは5000。
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """位置エンコーディングを入力に追加する。
        
        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: 位置エンコーディングが追加された出力 (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """セルフアテンションとフィードフォワードを持つ単一のTransformerブロック。
    
    セルフアテンション、フィードフォワードネットワーク、残差接続、
    事前層正規化を持つ単一のデコーダー層を実装する。
    
    Args:
        d_model (int): モデルの次元数。
        n_heads (int): アテンションヘッドの数。
        d_ff (int): フィードフォワード層の次元数。
        dropout (float): ドロップアウト確率。
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """Transformerブロックの順伝播を実行する。
        
        Args:
            x (torch.Tensor): 入力テンソル (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): アテンションマスク。Noneの場合は使用しない。
            
        Returns:
            torch.Tensor: Transformerブロックの出力 (batch_size, seq_len, d_model)
        """
        # Pre-LayerNorm: 残差接続付きセルフアテンション
        normalized_x = self.layer_norm1(x)
        attn_output = self.attn(normalized_x, mask)
        x = x + self.dropout(attn_output)
        
        # Pre-LayerNorm: 残差接続付きフィードフォワード
        normalized_x = self.layer_norm2(x)
        ff_output = self.mlp(normalized_x)
        x = x + self.dropout(ff_output)
        
        return x


class GPT2Model(nn.Module):
    """GPT-2モデルの実装。
    
    自己回帰言語モデリングのためのデコーダーのみのTransformerモデル。
    HuggingFace transformersの構造と命名規則に互換性を持つ。
    
    Args:
        vocab_size (int): 語彙サイズ。
        d_model (int): モデルの次元数。デフォルト: 384。
        n_heads (int): アテンションヘッドの数。デフォルト: 6。
        n_layers (int): Transformerレイヤーの数。デフォルト: 6。
        d_ff (int): フィードフォワード層の次元数。デフォルト: 1536。
        max_len (int): 最大シーケンス長。デフォルト: 512。
        dropout (float): ドロップアウト確率。デフォルト: 0.1。
    """
    
    def __init__(self, vocab_size, d_model=384, n_heads=6, n_layers=6,
                 d_ff=1536, max_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # トークンと位置埋め込み
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Transformerブロック
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 層正規化と出力射影
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 重みの初期化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """正規分布を使用して重みを初期化する。
        
        Args:
            module (nn.Module): 初期化するモジュール。
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len):
        """自己回帰生成用の因果（下三角）マスクを作成する。
        
        Args:
            seq_len (int): シーケンス長。
            
        Returns:
            torch.Tensor: 因果マスク (1, 1, seq_len, seq_len)
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # バッチとヘッド次元を追加
    
    def forward(self, input_ids, attention_mask=None):
        """GPT-2モデルの順伝播を実行する。
        
        Args:
            input_ids (torch.Tensor): 入力トークンID (batch_size, seq_len)
            attention_mask (torch.Tensor, optional): アテンションマスク。現在は未使用。
            
        Returns:
            torch.Tensor: 語彙に対するロジット (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.size()
        
        # 位置IDの作成
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # トークンと位置埋め込み
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        
        # 埋め込みの結合
        x = self.dropout(token_emb + pos_emb)
        
        # 因果マスクの作成
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Transformerブロックを通す
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # 最終層正規化
        x = self.layer_norm(x)
        
        # 語彙への出力射影
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """自己回帰サンプリングを使用してテキストを生成する。
        
        Args:
            input_ids (torch.Tensor): 入力トークンID (batch_size, seq_len)
            max_new_tokens (int): 生成する最大新規トークン数。デフォルトは50。
            temperature (float): サンプリング温度。デフォルトは1.0。
            top_k (int, optional): Top-kサンプリングのk値。Noneの場合は適用しない。
            
        Returns:
            torch.Tensor: 生成されたトークンID (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 現在のシーケンスの予測を取得
                logits = self.forward(generated)
                
                # 最後のトークンのロジットを取得
                next_token_logits = logits[:, -1, :] / temperature
                
                # 指定されていればTop-kフィルタリングを適用
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # 分布からサンプリング
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # シーケンスに追加
                generated = torch.cat([generated, next_token], dim=1)
                
                # 最大長を超えた場合は終了
                if generated.size(1) >= self.max_len:
                    break
        
        return generated
    
    def save_model(self, save_path, config=None, optimizer_state=None, epoch=None, loss=None):
        """モデルの重み、設定、訓練状態を保存する。
        
        Args:
            save_path (str): 保存先パス。
            config (dict, optional): 訓練設定。
            optimizer_state (dict, optional): オプティマイザーの状態。
            epoch (int, optional): エポック数。
            loss (float, optional): 損失値。
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'max_len': self.max_len,
                'n_heads': getattr(self, 'n_heads', None),
                'n_layers': len(self.transformer_blocks),
                'd_ff': getattr(self, 'd_ff', None)
            },
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__
        }
        
        if config is not None:
            save_dict['training_config'] = config
        if optimizer_state is not None:
            save_dict['optimizer_state_dict'] = optimizer_state
        if epoch is not None:
            save_dict['epoch'] = epoch
        if loss is not None:
            save_dict['loss'] = loss
            
        torch.save(save_dict, save_path)
        print(f"Model saved to: {save_path}")
    
    @classmethod
    def load_model(cls, load_path, device='cpu'):
        """保存されたチェックポイントからモデルを読み込む。
        
        Args:
            load_path (str): 読み込み元パス。
            device (str): 読み込み先デバイス。デフォルトは'cpu'。
            
        Returns:
            tuple: (モデルインスタンス, チェックポイント辞書)
            
        Raises:
            FileNotFoundError: モデルファイルが見つからない場合。
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        
        # モデル設定を抽出
        model_config = checkpoint['model_config']
        
        # モデルインスタンスを作成
        model = cls(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            max_len=model_config['max_len'],
            n_heads=model_config.get('n_heads', 12),
            n_layers=model_config.get('n_layers', 12),
            d_ff=model_config.get('d_ff', model_config['d_model'] * 4)
        )
        
        # モデル重みを読み込み
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"Model loaded from: {load_path}")
        print(f"Saved on: {checkpoint.get('timestamp', 'Unknown')}")
        
        return model, checkpoint
    
    def save_checkpoint(self, save_dir, epoch, optimizer, loss, config=None):
        """自動命名による訓練チェックポイントを保存する。
        
        Args:
            save_dir (str): 保存ディレクトリ。
            epoch (int): エポック数。
            optimizer (torch.optim.Optimizer): オプティマイザー。
            loss (float): 損失値。
            config (dict, optional): 設定辞書。
            
        Returns:
            str: 保存されたチェックポイントのパス。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"gpt2_checkpoint_epoch_{epoch}_{timestamp}.pt"
        checkpoint_path = os.path.join(save_dir, checkpoint_name)
        
        self.save_model(
            checkpoint_path,
            config=config,
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            loss=loss
        )
        
        return checkpoint_path


class TextDataset(Dataset):
    """訓練用のシンプルなテキストデータセット。

    Args:
        text (str): 訓練用テキスト。
        tokenizer: トークナイザー。
        max_length (int): 最大シーケンス長。デフォルトは1024。
        stride (int): シーケンス間のストライド。デフォルトはmax_lengthの半分。
    """

    def __init__(self, text, tokenizer, max_length=1024, stride=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length // 2

        print(f"Tokenizing text of length {len(text)} characters...")

        # テキスト全体をトークン化（進捗バー付き）
        print("Encoding text to tokens...")
        tokens = tokenizer.encode(text)
        print(f"Tokenized to {len(tokens)} tokens")

        # 重複するシーケンスを作成（進捗バー付き）
        self.sequences = []
        num_sequences = (len(tokens) - max_length) // self.stride + 1
        print(f"Creating {num_sequences} sequences...")
        
        for i in tqdm(range(0, len(tokens) - max_length + 1, self.stride), 
                     desc="Creating sequences", unit="sequence"):
            sequence = tokens[i:i + max_length]
            if len(sequence) == max_length:
                self.sequences.append(sequence)

        print(f"Created {len(self.sequences)} sequences of length {max_length}")
    
    def __len__(self):
        """データセットのサイズを返す。

        Returns:
            int: データセット内のシーケンス数。
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """指定されたインデックスのアイテムを取得する。

        Args:
            idx (int): アイテムのインデックス。

        Returns:
            torch.Tensor: トークン化されたシーケンス。
        """
        sequence = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.long)


def train_gpt2(model, dataloader, optimizer, device, num_epochs=5, save_dir=None, save_every=None):
    """チェックポイント保存機能付きのGPT-2モデル訓練ループ。

    Args:
        model (GPT2Model): 訓練するGPT-2モデル。
        dataloader (DataLoader): 訓練データローダー。
        optimizer (torch.optim.Optimizer): オプティマイザー。
        device (torch.device): 計算デバイス。
        num_epochs (int): 訓練エポック数。デフォルトは5。
        save_dir (str, optional): チェックポイント保存ディレクトリ。
        save_every (int, optional): チェックポイント保存間隔（ステップ単位）。

    Returns:
        str or None: 最終モデルのパス（save_dirが指定された場合）。
    """
    model.train()

    # 指定されていれば保存ディレクトリを作成
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {save_dir}")
        if save_every:
            print(f"Checkpoints will be saved every {save_every} steps")

    # 訓練状態ログファイルの初期化
    train_state_file = None
    if save_dir:
        train_state_file = os.path.join(save_dir, "train_state.json")
        # ファイルを初期化（空のリストで開始）
        with open(train_state_file, 'w') as f:
            json.dump([], f)
        print(f"Training state will be logged to: {train_state_file}")

    global_step = 0

    # エポック全体の進捗バー
    epoch_pbar = tqdm(range(num_epochs), desc="Epoch", position=0)

    for epoch in epoch_pbar:
        total_loss = 0
        num_batches = 0

        # バッチごとの進捗バー
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False)

        for batch_idx, batch in enumerate(batch_pbar):
            batch = batch.to(device)

            # 入力とターゲット（1位置シフト）を準備
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            # 順伝播
            logits = model(input_ids)

            # 損失計算
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            # 逆伝播
            optimizer.zero_grad()
            loss.backward()

            # 勾配のノルムを計算
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            # 訓練状態をtrain_state.jsonに記録
            if train_state_file:
                train_state_entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "timestamp": datetime.now().isoformat()
                }

                # 既存のデータを読み取り、新しいエントリを追加
                with open(train_state_file, 'r') as f:
                    train_states = json.load(f)
                train_states.append(train_state_entry)

                # ファイルに書き込み
                with open(train_state_file, 'w') as f:
                    json.dump(train_states, f, indent=2)

            # 進捗バーの説明を更新
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Step': global_step,
                'Avg Loss': f'{total_loss / num_batches:.4f}',
                'Grad Norm': f'{grad_norm.item():.4f}'
            })

            # ステップベースでチェックポイントを保存
            if save_dir and save_every and global_step % save_every == 0:
                checkpoint_name = f"checkpoint-{global_step}.pt"
                checkpoint_path = os.path.join(save_dir, checkpoint_name)

                model.save_model(
                    checkpoint_path,
                    config={'num_epochs': num_epochs, 'learning_rate': optimizer.param_groups[0]['lr']},
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch + 1,
                    loss=loss.item()
                )
                tqdm.write(f"Checkpoint saved at step {global_step}: {checkpoint_path}")

        avg_loss = total_loss / num_batches
        epoch_pbar.set_postfix({'Avg Loss': f'{avg_loss:.4f}'})
        tqdm.write(f'Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}')

    # 最終モデルを保存
    if save_dir:
        final_path = os.path.join(save_dir, "gpt2_final_model.pt")
        model.save_model(
            final_path,
            config={'num_epochs': num_epochs, 'learning_rate': optimizer.param_groups[0]['lr']},
            optimizer_state=optimizer.state_dict(),
            epoch=num_epochs,
            loss=avg_loss
        )
        print(f"Final model saved: {final_path}")
        return final_path

    return None


def load_wikipedia_dataset(num_samples=None):
    """Wikipediaデータセットを読み込む。

    Args:
        cache_dir (str, optional): データセットのキャッシュディレクトリ。
        num_samples (int, optional): 使用するサンプル数の制限。

    Returns:
        str: 結合されたテキストデータ。
    """
    print("Loading Wikipedia dataset...")

    # Wikipediaデータセットを読み込み
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train"
    )

    # 全件利用するためコメントアウト
    # if num_samples:
    #     ds = ds.select(range(min(num_samples, len(ds))))
    #     print(f"Using {len(ds)} samples from dataset")

    # テキストを結合（進捗バー付き）
    texts = []
    print(f"Processing {len(ds)} articles...")
    
    for item in tqdm(ds, desc="Processing articles", unit="article"):
        if item['text'] and len(item['text'].strip()) > 100:  # 短すぎるテキストをフィルタ
            item['text'] = item['text'] + "<|endoftext|>"
            texts.append(item['text'])

    print(f"Combining {len(texts)} articles...")
    combined_text = '\n\n'.join(texts)
    print(f"Loaded {len(texts)} articles, total length: {len(combined_text)} characters")

    return combined_text

# def load_c4_dataset():
#     """C4データセットを読み込む。

#     Args:
#         num_samples (int, optional): 使用するサンプル数の制限。

#     Returns:
#         str: 結合されたテキストデータ。
#     """
#     print("Loading C4 dataset...")

#     ds = load_dataset(
#         "allenai/c4",
#         "en",
#         split="train"
#     )

#     texts = []
#     for item in ds:
#         if item['text'] and len(item['text'].strip()) > 100:
#             texts.append(item['text'])

#     combined_text = '\n\n'.join(texts)
#     print(f"Loaded {len(texts)} samples, total length: {len(combined_text)} characters")

#     return combined_text


def parse_arguments():
    """コマンドライン引数を解析する。
    
    Returns:
        argparse.Namespace: 解析されたコマンドライン引数。
    """
    parser = argparse.ArgumentParser(description='GPT-2モデルの学習と生成')
    
    # モデルパラメータ
    parser.add_argument('--d_model', type=int, default=384, help='モデル次元 (default: 384)')
    parser.add_argument('--n_heads', type=int, default=6, help='アテンションヘッド数 (default: 6)')
    parser.add_argument('--n_layers', type=int, default=6, help='レイヤー数 (default: 6)')
    parser.add_argument('--d_ff', type=int, default=1536, help='フィードフォワード次元 (default: 1536)')
    parser.add_argument('--max_len', type=int, default=512, help='最大系列長 (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.1, help='ドロップアウト率 (default: 0.1)')
    
    # 学習パラメータ
    parser.add_argument('--batch_size', type=int, default=4, help='バッチサイズ (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='学習率 (default: 2e-4)')
    parser.add_argument('--num_epochs', type=int, default=1, help='エポック数 (default: 3)')
    
    # データセット関連
    parser.add_argument('--sequence_length', type=int, default=128, help='シーケンス長 (default: 128)')
    parser.add_argument('--num_samples', type=int, default=None, help='使用するサンプル数の制限 (default: None)')
    # その他
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='保存ディレクトリ (default: ./checkpoints)')
    parser.add_argument('--save_every', type=int, default=1000, help='チェックポイント保存間隔（ステップ単位） (default: 1000)')
    parser.add_argument('--prompt', type=str, default='The quick brown', help='生成用プロンプト')
    
    return parser.parse_args()

def main():
    """コマンドライン引数を受け取ってGPT-2モデルの訓練と生成を実行するメイン関数。
    
    コマンドライン引数に基づいてモデルを初期化し、訓練を実行し、
    最後にテキスト生成のテストを行う。
    """
    # コマンドライン引数を解析
    args = parse_arguments()
    
    # デバイスを設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # トークナイザーを初期化
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    
    # 引数から設定を構築
    config = {
        'vocab_size': vocab_size,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_ff,
        'max_len': args.max_len,
        'dropout': args.dropout
    }
    
    # モデルを初期化
    model = GPT2Model(**config).to(device)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # C4データセットを読み込み（サンプルテキストの代わり）
    dataset_text = load_wikipedia_dataset(num_samples=args.num_samples)
    
    # データセットとデータローダーを作成（RTX4090ノートPC向けに最適化）
    dataset = TextDataset(dataset_text, tokenizer, max_length=args.sequence_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # オプティマイザーを初期化（RTX4090ノートPC向けに調整）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    # チェックポイント保存付きでモデルを訓練
    print("Starting training...")
    final_model_path = train_gpt2(
        model, dataloader, optimizer, device,
        num_epochs=args.num_epochs, save_dir=args.save_dir, save_every=args.save_every
    )
    
    # テキスト生成をテスト
    print("\nTesting text generation...")
    model.eval()
    
    prompt = args.prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=30, temperature=0.8, top_k=40)
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
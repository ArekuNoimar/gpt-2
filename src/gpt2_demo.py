"""
GPT-2モデルデモスクリプト
GPT-2実装の様々な機能をデモンストレーションする
"""

import torch
import tiktoken
from gpt2_from_scratch import GPT2Model, TextDataset
from torch.utils.data import DataLoader


def demo_model_architecture():
    """モデルアーキテクチャとコンポーネントをデモンストレーションする。
    
    GPT-2モデルの構造、パラメータ数、および順伝播の動作を表示し、
    モデルの基本情報と機能を確認する。
    """
    print("=== GPT-2 Model Architecture Demo ===")
    
    # トークナイザーを初期化
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    
    # モデル設定
    config = {
        'vocab_size': vocab_size,
        'd_model': 384,
        'n_heads': 6,
        'n_layers': 6,
        'd_ff': 1536,
        'max_len': 512,
        'dropout': 0.1
    }
    
    # モデルを初期化
    model = GPT2Model(**config)
    
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Model dimension: {config['d_model']}")
    print(f"Number of attention heads: {config['n_heads']}")
    print(f"Number of layers: {config['n_layers']}")
    print(f"Feed-forward dimension: {config['d_ff']}")
    print(f"Maximum sequence length: {config['max_len']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 順伝播のテスト
    test_input = torch.randint(0, vocab_size, (2, 10))  # バッチサイズ2、シーケンス長10
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")
    print()


def demo_text_generation():
    """異なるパラメータでのテキスト生成をデモンストレーションする。
    
    複数のプロンプトと異なる生成設定（温度、top-kなど）を使用して、
    モデルのテキスト生成能力を評価し、設定の影響を比較する。
    """
    print("=== Text Generation Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 簡単なデモのための小さなモデルを初期化
    config = {
        'vocab_size': tokenizer.n_vocab,
        'd_model': 256,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 1024,
        'max_len': 256,
        'dropout': 0.1
    }
    
    model = GPT2Model(**config).to(device)
    
    # テストプロンプト
    prompts = [
        "The future of artificial intelligence",
        "In a world where technology",
        "Machine learning has revolutionized"
    ]
    
    # テストする生成パラメータ
    generation_configs = [
        {"temperature": 0.7, "top_k": 40, "max_new_tokens": 20},
        {"temperature": 1.0, "top_k": None, "max_new_tokens": 20},
        {"temperature": 0.5, "top_k": 10, "max_new_tokens": 20}
    ]
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        
        for i, gen_config in enumerate(generation_configs):
            with torch.no_grad():
                generated = model.generate(input_ids, **gen_config)
                generated_text = tokenizer.decode(generated[0].cpu().tolist())
            
            print(f"  Config {i+1} (temp={gen_config['temperature']}, top_k={gen_config['top_k']}): {generated_text}")
        print()


def demo_training_process():
    """メトリクス付きの訓練プロセスをデモンストレーションする。
    
    小さなモデルで簡単な訓練ループを実行し、訓練の進行状況、
    損失値の変化、および訓練プロセスの概要を表示する。
    """
    print("=== Training Process Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 簡単な訓練デモのための小さなモデル
    config = {
        'vocab_size': tokenizer.n_vocab,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'd_ff': 512,
        'max_len': 128,
        'dropout': 0.1
    }
    
    model = GPT2Model(**config).to(device)
    
    # サンプル訓練テキスト
    training_text = """
    Artificial intelligence is rapidly advancing. Machine learning algorithms can now perform tasks that were once thought impossible.
    Deep learning models have revolutionized computer vision, natural language processing, and many other fields.
    The transformer architecture introduced the concept of self-attention, which allows models to focus on relevant parts of the input.
    GPT models are large language models that generate text by predicting the next token in a sequence.
    """ * 20
    
    # データセットを作成
    dataset = TextDataset(training_text, tokenizer, max_length=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 訓練設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("Training for 2 epochs...")
    
    model.train()
    for epoch in range(2):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    print("Training completed!")
    print()


def demo_attention_visualization():
    """アテンション重みを抽出して可視化する方法をデモンストレーションする。
    
    テスト文を使用してモデルのアテンションパターンを調査し、
    アテンション重みの可視化に必要な情報を表示する。
    """
    print("=== Attention Visualization Demo ===")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # デモンストレーション用の小さなモデル
    config = {
        'vocab_size': tokenizer.n_vocab,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 512,
        'max_len': 64,
        'dropout': 0.0  # 一貫した結果のためドロップアウトなし
    }
    
    model = GPT2Model(**config)
    model.eval()
    
    # テスト文
    text = "The cat sat on the mat"
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens])
    
    print(f"Input text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded tokens: {[tokenizer.decode([t]) for t in tokens]}")
    
    # アテンションパターンを取得するための順伝播
    with torch.no_grad():
        # アテンション重みを返すためにモデルを修正する必要がある
        # 今のところはコンセプトを示すだけ
        logits = model(input_ids)
        
    print(f"Output logits shape: {logits.shape}")
    print("Note: To visualize attention, modify the model to return attention weights from each layer")
    print()


def main():
    """すべてのデモを実行する。
    
    GPT-2モデルの各機能を順次デモンストレーションし、
    モデルの完全な機能一覧を表示する。
    """
    print("GPT-2 Implementation Demo")
    print("=" * 50)
    
    demo_model_architecture()
    demo_text_generation() 
    demo_training_process()
    demo_attention_visualization()
    
    print("Demo completed! The GPT-2 implementation includes:")
    print("✓ Multi-head self-attention mechanism")
    print("✓ Transformer blocks with layer normalization")
    print("✓ Positional embeddings")
    print("✓ Causal masking for autoregressive generation")
    print("✓ Training loop with next-token prediction")
    print("✓ Text generation with temperature and top-k sampling")
    print("\nThe model is ready for training on larger datasets!")


if __name__ == "__main__":
    main()
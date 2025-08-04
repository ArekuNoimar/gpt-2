#!/usr/bin/env python3
"""
GPT-2推論スクリプト
スクラッチから訓練されたGPT-2モデルのためのシンプルな推論スクリプト。
チョックポイントディレクトリからモデルを読み込み、インタラクティブなテキスト生成を提供する。
"""

import torch
import tiktoken
import os
import sys
import argparse

# モジュールをインポートするためにprograms/srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gpt2_from_scratch import GPT2Model


def load_model_and_tokenizer(model_path, device='cpu'):
    """GPT-2モデルとトークナイザーを読み込む。
    
    指定されたパスからモデルのチェックポイントを読み込み、
    適切なデバイスに配置し、トークナイザーも初期化する。
    
    Args:
        model_path (str): モデルチェックポイントファイルのパス。
        device (str): モデルを読み込むデバイス。デフォルトは'cpu'。
        
    Returns:
        tuple: (GPT2Model, tokenizer, checkpoint) のタプル。
    """
    print(f"Loading model from: {model_path}")
    
    # クラスメソッドを使用してモデルを読み込み
    model, checkpoint = GPT2Model.load_model(model_path, device)
    model.eval()
    print(model)
    
    # tiktokenトークナイザーを初期化（訓練時と同じもの）
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print(f"Model loaded successfully!")
    print(f"Model configuration:")
    print(f"  - Vocabulary size: {checkpoint['model_config']['vocab_size']}")
    print(f"  - Model dimension: {checkpoint['model_config']['d_model']}")
    print(f"  - Number of layers: {checkpoint['model_config']['n_layers']}")
    print(f"  - Max sequence length: {checkpoint['model_config']['max_len']}")
    
    return model, tokenizer, checkpoint


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=40, device='cpu'):
    """読み込み済みモデルを使用してテキストを生成する。
    
    与えられたプロンプトから始めて、指定されたパラメータで
    テキストを自己回帰的に生成する。
    
    Args:
        model (GPT2Model): 訓練済みのGPT-2モデル。
        tokenizer: テキストエンコーディング用トークナイザー。
        prompt (str): 生成のための初期プロンプト。
        max_new_tokens (int): 生成する最大新規トークン数。デフォルトは50。
        temperature (float): サンプリング温度。デフォルトは0.8。
        top_k (int): Top-kサンプリングパラメータ。デフォルトは40。
        device (str): 推論を実行するデバイス。デフォルトは'cpu'。
        
    Returns:
        str: 生成されたテキスト。
    """
    # プロンプトをエンコード
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...")
    
    # テキストを生成
    with torch.no_grad():
        generated = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_k=top_k
        )
    
    # 生成されたテキストをデコード
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    
    return generated_text


def interactive_mode(model, tokenizer, device):
    """インタラクティブテキスト生成モード。
    
    ユーザーからの入力を受け取り、リアルタイムでテキスト生成を行う。
    生成パラメータの変更、ヘルプ表示などの機能を提供する。
    
    Args:
        model (GPT2Model): 訓練済みのGPT-2モデル。
        tokenizer: テキストエンコーディング用トークナイザー。
        device (torch.device): 推論を実行するデバイス。
    """
    print("\n" + "="*60)
    print("GPT-2 Interactive Text Generation")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for generation parameters")
    print("="*60)
    
    # デフォルト生成パラメータ
    max_tokens = 50
    temperature = 0.8
    top_k = 40
    
    while True:
        try:
            # ユーザー入力を取得
            prompt = input("\nEnter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if prompt.lower() == 'help':
                print(f"""
Current generation parameters:
  - max_tokens: {max_tokens} (number of new tokens to generate)
  - temperature: {temperature} (randomness, 0.1=conservative, 1.5=creative)
  - top_k: {top_k} (top-k sampling, None=disabled)

To change parameters, use:
  - 'set max_tokens 100'
  - 'set temperature 1.2'
  - 'set top_k 30'
                """)
                continue
            
            # パラメータ変更を処理
            if prompt.lower().startswith('set '):
                parts = prompt.split()
                if len(parts) == 3:
                    param, value = parts[1], parts[2]
                    try:
                        if param == 'max_tokens':
                            max_tokens = int(value)
                            print(f"Set max_tokens to {max_tokens}")
                        elif param == 'temperature':
                            temperature = float(value)
                            print(f"Set temperature to {temperature}")
                        elif param == 'top_k':
                            if value.lower() == 'none':
                                top_k = None
                                print("Disabled top_k sampling")
                            else:
                                top_k = int(value)
                                print(f"Set top_k to {top_k}")
                        else:
                            print(f"Unknown parameter: {param}")
                    except ValueError:
                        print(f"Invalid value for {param}: {value}")
                continue
            
            if not prompt:
                print("Please enter a prompt.")
                continue
            
            # テキストを生成
            generated_text = generate_text(
                model, tokenizer, prompt, max_tokens, temperature, top_k, device
            )
            
            print(f"\nGenerated text:\n{generated_text}")
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during generation: {e}")


def main():
    """メイン関数。
    
    コマンドライン引数を解析し、モデルを読み込み、
    プロンプトが指定されている場合は単一生成、
    そうでなければインタラクティブモードを開始する。
    
    Returns:
        int: 終了ステータスコード。
    """
    parser = argparse.ArgumentParser(description='GPT-2 Inference Script')
    parser.add_argument(
        '--model', '-m', 
        type=str, 
        default='programs/src/checkpoints/gpt2_final_model.pt',
        help='Path to the model checkpoint file'
    )
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        help='Text prompt for generation (if not provided, starts interactive mode)'
    )
    parser.add_argument(
        '--max_tokens', '-t',
        type=int,
        default=50,
        help='Maximum number of new tokens to generate (default: 50)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=40,
        help='Top-k sampling parameter (default: 40)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run inference on (default: auto)'
    )
    
    args = parser.parse_args()
    
    # デバイスを設定
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # モデルファイルの存在を確認
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print(f"Available models in checkpoints directory:")
        checkpoint_dir = os.path.dirname(args.model)
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pt'):
                    print(f"  - {os.path.join(checkpoint_dir, file)}")
        return 1
    
    try:
        # モデルとトークナイザーを読み込み
        model, tokenizer, checkpoint = load_model_and_tokenizer(args.model, device)
        
        if args.prompt:
            # 単一生成モード
            generated_text = generate_text(
                model, tokenizer, args.prompt, 
                args.max_tokens, args.temperature, args.top_k, device
            )
            print(f"\nGenerated text:\n{generated_text}")
        else:
            # インタラクティブモード
            interactive_mode(model, tokenizer, device)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
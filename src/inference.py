#!/usr/bin/env python3
"""
GPT-2 Inference Script
Simple inference script for GPT-2 models trained from scratch.
Loads model from checkpoints directory and provides interactive text generation.
"""

import torch
import tiktoken
import os
import sys
import argparse

# Add the programs/src directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gpt2_from_scratch import GPT2Model


def load_model_and_tokenizer(model_path, device='cpu'):
    """Load the GPT-2 model and tokenizer"""
    print(f"Loading model from: {model_path}")
    
    # Load the model using the class method
    model, checkpoint = GPT2Model.load_model(model_path, device)
    model.eval()
    
    # Initialize tiktoken tokenizer (same as used during training)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    print(f"Model loaded successfully!")
    print(f"Model configuration:")
    print(f"  - Vocabulary size: {checkpoint['model_config']['vocab_size']}")
    print(f"  - Model dimension: {checkpoint['model_config']['d_model']}")
    print(f"  - Number of layers: {checkpoint['model_config']['n_layers']}")
    print(f"  - Max sequence length: {checkpoint['model_config']['max_len']}")
    
    return model, tokenizer, checkpoint


def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, top_k=40, device='cpu'):
    """Generate text using the loaded model"""
    # Encode the prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...")
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_k=top_k
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    
    return generated_text


def interactive_mode(model, tokenizer, device):
    """Interactive text generation mode"""
    print("\n" + "="*60)
    print("GPT-2 Interactive Text Generation")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for generation parameters")
    print("="*60)
    
    # Default generation parameters
    max_tokens = 50
    temperature = 0.8
    top_k = 40
    
    while True:
        try:
            # Get user input
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
            
            # Handle parameter changes
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
            
            # Generate text
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
    """Main function"""
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
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if model file exists
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
        # Load model and tokenizer
        model, tokenizer, checkpoint = load_model_and_tokenizer(args.model, device)
        
        if args.prompt:
            # Single generation mode
            generated_text = generate_text(
                model, tokenizer, args.prompt, 
                args.max_tokens, args.temperature, args.top_k, device
            )
            print(f"\nGenerated text:\n{generated_text}")
        else:
            # Interactive mode
            interactive_mode(model, tokenizer, device)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
"""
GPT-2 Model Implementation from Scratch
Based on the architecture described in "Attention Is All You Need" and OpenAI's GPT-2 paper
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


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V), attention_weights
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and split into heads
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear projection
        output = self.w_o(attn_output)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed Forward Network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Positional encoding using sinusoidal functions"""
    
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
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and feed-forward"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x


class GPT2Model(nn.Module):
    """GPT-2 Model Implementation"""
    
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=12, 
                 d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Layer normalization and output projection
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using normal distribution"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len):
        """Create causal (lower triangular) mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Token and position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len).to(input_ids.device)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Output projection to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Generate text using autoregressive sampling"""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get predictions for the current sequence
                logits = self.forward(generated)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Break if we exceed max length
                if generated.size(1) >= self.max_len:
                    break
        
        return generated
    
    def save_model(self, save_path, config=None, optimizer_state=None, epoch=None, loss=None):
        """Save model weights, configuration, and training state"""
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
        """Load model from saved checkpoint"""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        
        # Extract model configuration
        model_config = checkpoint['model_config']
        
        # Create model instance
        model = cls(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            max_len=model_config['max_len'],
            n_heads=model_config.get('n_heads', 12),
            n_layers=model_config.get('n_layers', 12),
            d_ff=model_config.get('d_ff', model_config['d_model'] * 4)
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"Model loaded from: {load_path}")
        print(f"Saved on: {checkpoint.get('timestamp', 'Unknown')}")
        
        return model, checkpoint
    
    def save_checkpoint(self, save_dir, epoch, optimizer, loss, config=None):
        """Save training checkpoint with automatic naming"""
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
    """Simple text dataset for training"""
    
    def __init__(self, text, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize the entire text
        tokens = tokenizer.encode(text)
        
        # Create overlapping sequences
        self.sequences = []
        for i in range(0, len(tokens) - max_length, max_length // 2):
            sequence = tokens[i:i + max_length]
            if len(sequence) == max_length:
                self.sequences.append(sequence)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.long)


def train_gpt2(model, dataloader, optimizer, device, num_epochs=5, save_dir=None, save_every=None):
    """Training loop for GPT-2 model with checkpoint saving"""
    model.train()
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"Checkpoints will be saved to: {save_dir}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Prepare input and target (shifted by one position)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            # Forward pass
            logits = model(input_ids)
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                targets.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint if specified
        if save_dir and save_every and (epoch + 1) % save_every == 0:
            checkpoint_path = model.save_checkpoint(
                save_dir, 
                epoch + 1, 
                optimizer, 
                avg_loss,
                config={'num_epochs': num_epochs, 'learning_rate': optimizer.param_groups[0]['lr']}
            )
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
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


def main():
    """Main function to demonstrate GPT-2 usage"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    
    # Model configuration (small GPT-2 for demonstration)
    config = {
        'vocab_size': vocab_size,
        'd_model': 384,
        'n_heads': 6,
        'n_layers': 6,
        'd_ff': 1536,
        'max_len': 512,
        'dropout': 0.1
    }
    
    # Initialize model
    model = GPT2Model(**config).to(device)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Sample text for training (you can replace this with your own dataset)
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for training our GPT-2 model.
    Machine learning is fascinating. Deep learning models like transformers have revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant parts of the input sequence.
    Natural language processing has made significant advances in recent years. Large language models can generate coherent text.
    Transformer architectures use self-attention to process sequences. The attention mechanism allows models to focus on relevant information.
    GPT models are autoregressive language models that generate text one token at a time. They use causal masking during training.
    The training process involves predicting the next token given the previous tokens. This is called next-token prediction.
    Language models learn patterns in text data through unsupervised learning. They capture statistical regularities in language.
    Deep learning has transformed many fields including computer vision and natural language processing.
    Neural networks with many layers can learn complex representations of data.
    """ * 10  # Repeat the text to make it longer
    
    # Create dataset and dataloader
    dataset = TextDataset(sample_text, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Train the model with checkpoint saving
    print("Starting training...")
    save_dir = "./checkpoints"
    final_model_path = train_gpt2(
        model, dataloader, optimizer, device, 
        num_epochs=3, save_dir=save_dir, save_every=1
    )
    
    # Test text generation
    print("\nTesting text generation...")
    model.eval()
    
    prompt = "The quick brown"
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=30, temperature=0.8, top_k=40)
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
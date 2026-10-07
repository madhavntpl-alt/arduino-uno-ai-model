#!/usr/bin/env python3
"""
evaluate.py — Model evaluation and generation testing for UNO-MIND

Evaluates model quality: perplexity, accuracy, generation samples.
Tests both float and quantized models.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Run: pip install torch")
    sys.exit(1)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ByteLevelTokenizer:
    """Byte-level tokenizer matching Arduino implementation."""
    
    def __init__(self):
        self.vocab_size = 256
        self.bos_id = 255
        self.eos_id = 254
    
    def encode(self, text: str) -> list:
        tokens = [self.bos_id]
        tokens.extend(text.encode('utf-8'))
        tokens.append(self.eos_id)
        return tokens
    
    def decode(self, tokens: list) -> str:
        bytes_list = [t for t in tokens if t < 254]
        return bytes(bytes_list).decode('utf-8', errors='replace')
    
    def decode_token(self, token: int) -> str:
        if token < 254:
            return bytes([token]).decode('utf-8', errors='replace')
        return ''


class TextDataset(torch.utils.data.Dataset):
    """Dataset for evaluation."""
    
    def __init__(self, file_path: Path, context_window: int, bos_id: int, eos_id: int):
        self.context_window = context_window
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        text = file_path.read_text(encoding='utf-8')
        lines = [ln.strip() for ln in text.strip().split('\n') if ln.strip()]
        
        all_tokens = []
        for line in lines:
            tokens = [bos_id] + list(line.encode('utf-8')) + [eos_id]
            all_tokens.extend(tokens)
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        
        # Create examples (stride=1 for full eval)
        seq_len = context_window + 1
        self.examples = []
        for i in range(0, len(self.tokens) - seq_len + 1):
            self.examples.append(self.tokens[i:i + seq_len])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        window = self.examples[idx]
        return window[:-1], window[-1]


class UnoMindModel(nn.Module):
    """Model definition (must match train_model.py)."""
    
    def __init__(self, vocab_size: int, embed_dim: int, context_window: int, 
                 hidden_dim: int, tie_weights: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_window = context_window
        self.hidden_dim = hidden_dim
        self.input_dim = context_window * embed_dim
        self.tie_weights = tie_weights
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(self.input_dim, hidden_dim, bias=True)
        
        if tie_weights:
            self.output = nn.Linear(hidden_dim, vocab_size, bias=True)
            if hidden_dim != embed_dim:
                self.tie_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        else:
            self.output = nn.Linear(hidden_dim, vocab_size, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        if not self.tie_weights:
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            nn.init.zeros_(self.output.bias)
            if hasattr(self, 'tie_proj'):
                nn.init.xavier_uniform_(self.tie_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x.view(batch_size, -1)
        x = self.hidden(x)
        x = F.relu(x)
        
        if self.tie_weights:
            if hasattr(self, 'tie_proj'):
                x = self.tie_proj(x)
            logits = F.linear(x, self.embedding.weight, self.output.bias)
        else:
            logits = self.output(x)
        return logits


def evaluate_perplexity(model: nn.Module, dataloader, device: torch.device) -> dict:
    """Compute perplexity and accuracy on validation set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for contexts, targets in dataloader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            
            logits = model(contexts)
            loss = criterion(logits, targets)
            
            total_loss += loss.item() * contexts.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_tokens += contexts.size(0)
    
    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
    }


def evaluate_generation(model: nn.Module, tokenizer: ByteLevelTokenizer, 
                        prompts: list, device: torch.device, 
                        max_len: int = 100, temperature: float = 1.0) -> list:
    """Generate samples for qualitative evaluation."""
    model.eval()
    results = []
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        context_window = model.context_window
        
        with torch.no_grad():
            for _ in range(max_len):
                context = tokens[-context_window:]
                if len(context) < context_window:
                    context = [tokenizer.bos_id] * (context_window - len(context)) + context
                
                input_tensor = torch.tensor([context], dtype=torch.long, device=device)
                logits = model(input_tensor)[0]
                
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Greedy
                next_token = logits.argmax().item()
                tokens.append(next_token)
                
                if next_token == tokenizer.eos_id:
                    break
        
        generated = tokenizer.decode(tokens)
        # Remove prompt from output for cleaner display
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        
        results.append({
            'prompt': prompt,
            'generated': generated.strip(),
            'full': tokenizer.decode(tokens)
        })
    
    return results


def evaluate_repetition(model: nn.Module, tokenizer: ByteLevelTokenizer,
                        device: torch.device, num_samples: int = 50) -> dict:
    """Measure repetition rate in generations."""
    model.eval()
    
    prompts = [
        "hello", "what is", "the", "i think", "robotics is",
        "arduino", "gravity", "computer", "science", "explain"
    ]
    
    total_tokens = 0
    repeated_bigrams = 0
    repeated_trigrams = 0
    eos_count = 0
    max_len_reached = 0
    
    with torch.no_grad():
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            tokens = tokenizer.encode(prompt)
            context_window = model.context_window
            generated_tokens = []
            
            for _ in range(128):
                context = tokens[-context_window:]
                if len(context) < context_window:
                    context = [tokenizer.bos_id] * (context_window - len(context)) + context
                
                input_tensor = torch.tensor([context], dtype=torch.long, device=device)
                logits = model(input_tensor)[0]
                next_token = logits.argmax().item()
                tokens.append(next_token)
                generated_tokens.append(next_token)
                
                if next_token == tokenizer.eos_id:
                    eos_count += 1
                    break
            else:
                max_len_reached += 1
            
            # Check repetition in generated portion
            for n in [2, 3]:
                ngrams = [tuple(generated_tokens[j:j+n]) 
                         for j in range(len(generated_tokens) - n + 1)]
                if ngrams:
                    unique = set(ngrams)
                    if n == 2:
                        repeated_bigrams += len(ngrams) - len(unique)
                    else:
                        repeated_trigrams += len(ngrams) - len(unique)
            
            total_tokens += len(generated_tokens)
    
    return {
        'total_generated_tokens': total_tokens,
        'repetition_rate_bigram': repeated_bigrams / max(1, total_tokens),
        'repetition_rate_trigram': repeated_trigrams / max(1, total_tokens),
        'eos_rate': eos_count / num_samples,
        'max_len_rate': max_len_reached / num_samples,
        'avg_length': total_tokens / num_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate UNO-MIND model")
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--model', help='Model checkpoint (.pt)')
    parser.add_argument('--quantized', help='Quantized model (.npz) for comparison')
    parser.add_argument('--val-file', help='Validation data file')
    parser.add_argument('--prompts', nargs='+', help='Custom prompts for generation')
    parser.add_argument('--max-len', type=int, default=100, help='Max generation length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    mcfg = config['model']
    tokenizer = ByteLevelTokenizer()
    
    # Default prompts
    test_prompts = args.prompts or [
        "hello",
        "what is robotics",
        "what is arduino",
        "what is gravity",
        "explain electricity",
        "what is a computer",
        "tell me about science",
        "how are you",
        "define robot",
        "the earth is",
    ]
    
    # Load float model
    model_path = Path(args.model) if args.model else Path("uno_mind_model.pt")
    if model_path.exists():
        print(f"\n=== Evaluating Float Model: {model_path} ===")
        checkpoint = torch.load(model_path, map_location=device)
        
        model = UnoMindModel(
            vocab_size=mcfg['vocab_size'],
            embed_dim=mcfg['embed_dim'],
            context_window=mcfg['context_window'],
            hidden_dim=mcfg['hidden_dim'],
            tie_weights=mcfg['tie_weights']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Validation perplexity
        val_file = Path(args.val_file) if args.val_file else Path(config['dataset']['val_file'])
        if val_file.exists():
            val_dataset = TextDataset(val_file, mcfg['context_window'], 
                                     tokenizer.bos_id, tokenizer.eos_id)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
            
            metrics = evaluate_perplexity(model, val_loader, device)
            print(f"Validation Loss: {metrics['loss']:.4f}")
            print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
            print(f"Validation Perplexity: {metrics['perplexity']:.2f}")
        
        # Generation samples
        print("\n--- Generation Samples (Float) ---")
        samples = evaluate_generation(model, tokenizer, test_prompts, device, 
                                     args.max_len, args.temperature)
        for s in samples:
            print(f"> {s['prompt']}")
            print(f"  {s['generated'][:200]}")
            print()
        
        # Repetition analysis
        rep = evaluate_repetition(model, tokenizer, device)
        print("--- Repetition Analysis ---")
        print(f"Avg length: {rep['avg_length']:.1f} tokens")
        print(f"EOS rate: {rep['eos_rate']*100:.1f}%")
        print(f"Max len rate: {rep['max_len_rate']*100:.1f}%")
        print(f"Bigram repetition: {rep['repetition_rate_bigram']*100:.1f}%")
        print(f"Trigram repetition: {rep['repetition_rate_trigram']*100:.1f}%")
    
    # Load and evaluate quantized model (simulated)
    quantized_path = Path(args.quantized) if args.quantized else Path("uno_mind_quantized.npz")
    if quantized_path.exists():
        print(f"\n=== Evaluating Quantized Model (Simulated): {quantized_path} ===")
        data = np.load(quantized_path, allow_pickle=True)
        
        class QModel:
            pass
        qmodel = QModel()
        for key in data.files:
            setattr(qmodel, key, data[key])
        
        # Simulated quantized forward
        def quantized_forward(contexts):
            batch_size = contexts.size(0)
            # Dequantize
            emb_w = torch.from_numpy(qmodel.embedding_weight).float() * qmodel.embedding_scale
            hid_w = torch.from_numpy(qmodel.hidden_weight).float() * qmodel.hidden_scale
            hid_b = torch.from_numpy(qmodel.hidden_bias).float() * qmodel.hidden_bias_scale
            
            if qmodel.tie_weights:
                out_w = emb_w.t()
            else:
                out_w = torch.from_numpy(qmodel.output_weight).float() * qmodel.output_scale
            out_b = torch.from_numpy(qmodel.output_bias).float() * qmodel.output_bias_scale
            
            # Forward
            x = F.embedding(contexts, emb_w.to(device))
            x = x.view(batch_size, -1)
            x = F.linear(x, hid_w.to(device), hid_b.to(device))
            x = F.relu(x)
            logits = F.linear(x, out_w.to(device), out_b.to(device))
            return logits
        
        # Quick validation
        val_file = Path(args.val_file) if args.val_file else Path(config['dataset']['val_file'])
        if val_file.exists():
            val_dataset = TextDataset(val_file, mcfg['context_window'], 
                                     tokenizer.bos_id, tokenizer.eos_id)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
            
            criterion = nn.CrossEntropyLoss()
            total_loss = 0.0
            total_correct = 0
            total_tokens = 0
            
            with torch.no_grad():
                for i, (contexts, targets) in enumerate(val_loader):
                    contexts = contexts.to(device)
                    targets = targets.to(device)
                    
                    logits = quantized_forward(contexts)
                    loss = criterion(logits, targets)
                    
                    total_loss += loss.item() * contexts.size(0)
                    preds = logits.argmax(dim=-1)
                    total_correct += (preds == targets).sum().item()
                    total_tokens += contexts.size(0)
                    
                    if i >= 10:  # Quick eval
                        break
            
            avg_loss = total_loss / total_tokens
            accuracy = total_correct / total_tokens
            perplexity = math.exp(min(avg_loss, 20))
            
            print(f"Quantized Validation Loss: {avg_loss:.4f}")
            print(f"Quantized Accuracy: {accuracy:.4f}")
            print(f"Quantized Perplexity: {perplexity:.2f}")
        
        # Generation with quantized
        print("\n--- Generation Samples (Quantized Sim) ---")
        for prompt in test_prompts[:5]:
            tokens = tokenizer.encode(prompt)
            for _ in range(args.max_len):
                context = tokens[-mcfg['context_window']:]
                if len(context) < mcfg['context_window']:
                    context = [tokenizer.bos_id] * (mcfg['context_window'] - len(context)) + context
                
                input_tensor = torch.tensor([context], dtype=torch.long, device=device)
                logits = quantized_forward(input_tensor)[0]
                
                if args.temperature != 1.0:
                    logits = logits / args.temperature
                
                next_token = logits.argmax().item()
                tokens.append(next_token)
                
                if next_token == tokenizer.eos_id:
                    break
            
            generated = tokenizer.decode(tokens)
            if generated.startswith(prompt):
                generated = generated[len(prompt):]
            print(f"> {prompt}")
            print(f"  {generated[:200]}")
            print()


if __name__ == "__main__":
    main()
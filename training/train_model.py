#!/usr/bin/env python3
"""
train_model.py — Model training for UNO-MIND

Trains a compact causal language model (fixed-context MLP) on next-token prediction.
Supports Quantization-Aware Training (QAT) for INT8 deployment.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import math
import time

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


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class TextDataset(Dataset):
    """Dataset for next-token prediction from tokenized text."""
    
    def __init__(self, file_path: Path, context_window: int, bos_id: int, eos_id: int, stride: int = 1):
        self.context_window = context_window
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        print(f"Loading data from {file_path}...")
        text = file_path.read_text(encoding='utf-8')
        lines = [ln.strip() for ln in text.strip().split('\n') if ln.strip()]
        
        # Tokenize all lines (byte-level)
        all_tokens = []
        for line in lines:
            tokens = [bos_id] + list(line.encode('utf-8')) + [eos_id]
            all_tokens.extend(tokens)
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.tokens)}")
        
        # Create sliding windows
        self.examples = []
        seq_len = context_window + 1  # +1 for target
        for i in range(0, len(self.tokens) - seq_len + 1, stride):
            window = self.tokens[i:i + seq_len]
            self.examples.append(window)
        
        print(f"Training examples: {len(self.examples)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        window = self.examples[idx]
        return window[:-1], window[-1]  # (context, target)


class UnoMindModel(nn.Module):
    """
    Compact fixed-context MLP language model.
    
    Architecture:
    - Embedding: vocab_size x embed_dim
    - Flatten: context_window * embed_dim -> input_dim
    - Hidden: input_dim -> hidden_dim (ReLU)
    - Output: hidden_dim -> vocab_size
    
    Optional: tie embedding and output weights
    """
    
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
            # Output projection uses transposed embedding weights
            self.output = nn.Linear(hidden_dim, vocab_size, bias=True)
            # Tie weights: output.weight = embedding.weight.T
            # Note: This requires hidden_dim == embed_dim for perfect tying
            # We'll use a projection if dims differ
            if hidden_dim != embed_dim:
                self.tie_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        else:
            self.output = nn.Linear(hidden_dim, vocab_size, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        # Initialize hidden layer
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        # Initialize output layer
        if not self.tie_weights:
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            nn.init.zeros_(self.output.bias)
            if hasattr(self, 'tie_proj'):
                nn.init.xavier_uniform_(self.tie_proj.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, context_window]
        batch_size = x.size(0)
        
        # Embedding lookup
        x = self.embedding(x)  # [batch, context_window, embed_dim]
        
        # Flatten context
        x = x.view(batch_size, -1)  # [batch, input_dim]
        
        # Hidden layer + ReLU
        x = self.hidden(x)
        x = F.relu(x)
        
        # Output projection
        if self.tie_weights:
            if hasattr(self, 'tie_proj'):
                x = self.tie_proj(x)  # [batch, embed_dim]
            # Use embedding weights transposed
            logits = F.linear(x, self.embedding.weight, self.output.bias)
        else:
            logits = self.output(x)
        
        return logits
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def prepare_qat_model(model: nn.Module) -> nn.Module:
    """Prepare model for Quantization-Aware Training."""
    # Use PyTorch's QAT API
    model.train()
    # Fuse modules for better quantization
    # Note: For custom model, we use fake quantization
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare_qat(model, inplace=False)
    return model_prepared


def convert_qat_model(model: nn.Module) -> nn.Module:
    """Convert QAT model to quantized INT8 model."""
    model.eval()
    model_quantized = torch.quantization.convert(model, inplace=False)
    return model_quantized


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device, 
                   criterion) -> tuple:
    """Evaluate model on validation set."""
    model.eval()
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
    perplexity = math.exp(min(avg_loss, 20))  # Cap for stability
    
    return avg_loss, accuracy, perplexity


def generate_sample(model: nn.Module, tokenizer, prompt: str, max_len: int, 
                    device: torch.device, temperature: float = 1.0) -> str:
    """Generate text sample for qualitative evaluation."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    context_window = model.context_window
    
    with torch.no_grad():
        for _ in range(max_len):
            # Take last context_window tokens
            context = tokens[-context_window:]
            if len(context) < context_window:
                context = [tokenizer.bos_id] * (context_window - len(context)) + context
            
            input_tensor = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_tensor)[0]  # [vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Greedy (argmax) for now
            next_token = logits.argmax().item()
            tokens.append(next_token)
            
            if next_token == tokenizer.eos_id:
                break
    
    return tokenizer.decode(tokens)


def main():
    parser = argparse.ArgumentParser(description="Train UNO-MIND model")
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--train-file', help='Training data file')
    parser.add_argument('--val-file', help='Validation data file')
    parser.add_argument('--output', help='Output model checkpoint')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate')
    parser.add_argument('--generate', help='Generate sample from prompt')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device(config['training']['device'])
    print(f"Using device: {device}")
    
    # Model config
    mcfg = config['model']
    tcfg = config['training']
    
    # Tokenizer config (byte-level)
    bos_id = config['tokenizer']['special_tokens']['bos']
    eos_id = config['tokenizer']['special_tokens']['eos']
    
    # Create model
    model = UnoMindModel(
        vocab_size=mcfg['vocab_size'],
        embed_dim=mcfg['embed_dim'],
        context_window=mcfg['context_window'],
        hidden_dim=mcfg['hidden_dim'],
        tie_weights=mcfg['tie_weights']
    ).to(device)
    
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Architecture: vocab={mcfg['vocab_size']}, embed={mcfg['embed_dim']}, "
          f"ctx={mcfg['context_window']}, hidden={mcfg['hidden_dim']}")
    
    # Resume if requested
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # QAT setup
    qat_enabled = tcfg['qat']['enabled']
    qat_model = None
    if qat_enabled:
        print("Preparing Quantization-Aware Training...")
        qat_model = prepare_qat_model(model)
        qat_model.to(device)
    
    # Datasets
    train_file = Path(args.train_file) if args.train_file else Path(config['dataset']['train_file'])
    val_file = Path(args.val_file) if args.val_file else Path(config['dataset']['val_file'])
    
    train_dataset = TextDataset(train_file, mcfg['context_window'], bos_id, eos_id, stride=4)
    val_dataset = TextDataset(val_file, mcfg['context_window'], bos_id, eos_id, stride=4)
    
    train_loader = DataLoader(
        train_dataset, batch_size=tcfg['batch_size'], shuffle=True,
        num_workers=tcfg['num_workers'], pin_memory=tcfg['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=tcfg['batch_size'], shuffle=False,
        num_workers=tcfg['num_workers'], pin_memory=tcfg['pin_memory']
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg['label_smoothing'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg['learning_rate'], 
                                   weight_decay=tcfg['weight_decay'])
    
    # LR scheduler
    if tcfg['lr_schedule'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg['epochs'])
    elif tcfg['lr_schedule'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None
    
    # Training loop
    best_val_loss = float('inf')
    output_path = Path(args.output) if args.output else Path("uno_mind_model.pt")
    
    if args.eval_only:
        val_loss, val_acc, val_ppl = evaluate_model(model, val_loader, device, criterion)
        print(f"Validation: Loss={val_loss:.4f}, Acc={val_acc:.4f}, PPL={val_ppl:.2f}")
        if args.generate:
            # Load tokenizer
            from train_tokenizer import ByteLevelTokenizer
            tokenizer = ByteLevelTokenizer()
            sample = generate_sample(model, tokenizer, args.generate, 100, device)
            print(f"\nPrompt: {args.generate}")
            print(f"Generated: {sample}")
        return
    
    print(f"\nStarting training for {tcfg['epochs']} epochs...")
    
    for epoch in range(start_epoch, tcfg['epochs']):
        epoch_start = time.time()
        
        # Switch to QAT model after specified epoch
        current_model = qat_model if (qat_enabled and epoch >= tcfg['qat']['start_epoch']) else model
        current_model.train()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch_idx, (contexts, targets) in enumerate(train_loader):
            contexts = contexts.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            logits = current_model(contexts)
            loss = criterion(logits, targets)
            loss.backward()
            
            if tcfg['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(current_model.parameters(), tcfg['grad_clip'])
            
            optimizer.step()
            
            total_loss += loss.item() * contexts.size(0)
            total_tokens += contexts.size(0)
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{tcfg['epochs']} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / total_tokens
        
        # Validation
        eval_model = qat_model if (qat_enabled and epoch >= tcfg['qat']['start_epoch']) else model
        val_loss, val_acc, val_ppl = evaluate_model(eval_model, val_loader, device, criterion)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{tcfg['epochs']} ({epoch_time:.1f}s): "
              f"Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, Val PPL={val_ppl:.2f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model = qat_model if (qat_enabled and epoch >= tcfg['qat']['start_epoch']) else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, output_path)
            print(f"  -> Saved best model to {output_path}")
        
        if scheduler:
            scheduler.step()
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    
    # Generate sample
    from train_tokenizer import ByteLevelTokenizer
    tokenizer = ByteLevelTokenizer()
    sample = generate_sample(model, tokenizer, "what is ", 50, device)
    print(f"\nSample generation: 'what is ' -> {sample}")


if __name__ == "__main__":
    main()
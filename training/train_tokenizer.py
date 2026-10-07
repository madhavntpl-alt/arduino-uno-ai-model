#!/usr/bin/env python3
"""
train_tokenizer.py — Tokenizer training for UNO-MIND

For byte-level tokenizer: no training needed, just creates vocab mapping.
For BPE: trains BPE merges on the dataset (optional future upgrade).
"""

import argparse
import yaml
import json
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ByteLevelTokenizer:
    """Byte-level tokenizer: 256 tokens (0-253 data, 254=EOS, 255=BOS)."""
    
    def __init__(self):
        self.vocab_size = 256
        self.bos_id = 255
        self.eos_id = 254
        # No UNK needed
        
    def encode(self, text: str) -> list:
        """Encode string to token IDs (bytes + BOS/EOS)."""
        tokens = [self.bos_id]
        tokens.extend(text.encode('utf-8'))
        tokens.append(self.eos_id)
        return tokens
    
    def decode(self, tokens: list) -> str:
        """Decode token IDs to string (skip special tokens)."""
        bytes_list = [t for t in tokens if t < 254]
        return bytes(bytes_list).decode('utf-8', errors='replace')
    
    def decode_token(self, token: int) -> str:
        """Decode single token for streaming output."""
        if token < 254:
            return bytes([token]).decode('utf-8', errors='replace')
        return ''  # Skip BOS/EOS
    
    def save(self, path: Path):
        """Save tokenizer config (mostly for compatibility)."""
        data = {
            'type': 'byte_level',
            'vocab_size': self.vocab_size,
            'bos_id': self.bos_id,
            'eos_id': self.eos_id,
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"Tokenizer saved to {path}")


class BPETokenizer:
    """BPE tokenizer (optional - for future upgrade to subword)."""
    
    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size
        self.merges = {}  # (byte1, byte2) -> new_token
        self.vocab = {}   # token -> bytes
        
    def train(self, texts: list, vocab_size: int = None):
        """Train BPE on list of strings. Simplified implementation."""
        if vocab_size:
            self.vocab_size = vocab_size
        
        # Initialize with byte vocab (0-255)
        for i in range(256):
            self.vocab[i] = bytes([i])
        
        # Count pairs
        from collections import Counter
        
        # Convert texts to byte sequences
        sequences = [list(t.encode('utf-8')) for t in texts]
        
        num_merges = self.vocab_size - 256
        print(f"Training BPE: {num_merges} merges...")
        
        for merge_idx in range(num_merges):
            # Count all adjacent pairs
            pair_counts = Counter()
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i+1])] += 1
            
            if not pair_counts:
                break
                
            # Most frequent pair
            best_pair, count = pair_counts.most_common(1)[0]
            if count < 2:
                break
            
            new_token = 256 + merge_idx
            self.merges[best_pair] = new_token
            self.vocab[new_token] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Apply merge to all sequences
            new_sequences = []
            for seq in sequences:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i+1]) == best_pair:
                        new_seq.append(new_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_sequences.append(new_seq)
            sequences = new_sequences
            
            if merge_idx % 50 == 0:
                print(f"  Merge {merge_idx}/{num_merges}: {best_pair} -> {new_token} (count={count})")
        
        print(f"BPE training complete. Final vocab: {len(self.vocab)}")
    
    def encode(self, text: str) -> list:
        """Encode using learned BPE merges."""
        tokens = list(text.encode('utf-8'))
        # Apply merges in order
        for (b1, b2), new_token in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == b1 and tokens[i+1] == b2:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def decode(self, tokens: list) -> str:
        bytes_list = []
        for t in tokens:
            if t in self.vocab:
                bytes_list.extend(self.vocab[t])
        return bytes(bytes_list).decode('utf-8', errors='replace')
    
    def save(self, path: Path):
        data = {
            'type': 'bpe',
            'vocab_size': self.vocab_size,
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            'vocab': {str(k): list(v) for k, v in self.vocab.items()},
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"BPE tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        tok = cls(data['vocab_size'])
        tok.merges = {tuple(map(int, k.split(','))): v for k, v in data['merges'].items()}
        tok.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()}
        return tok


def main():
    parser = argparse.ArgumentParser(description="Train tokenizer for UNO-MIND")
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--train-file', help='Training data file')
    parser.add_argument('--output', help='Output tokenizer file')
    parser.add_argument('--type', choices=['byte_level', 'bpe'], help='Override tokenizer type')
    args = parser.parse_args()
    
    config = load_config(args.config)
    tokenizer_type = args.type or config['tokenizer']['type']
    
    train_file = Path(args.train_file) if args.train_file else Path(config['dataset']['train_file'])
    output_path = Path(args.output) if args.output else Path(config['tokenizer'].get('save_path', 'tokenizer.json'))
    
    if not train_file.exists():
        print(f"Training file not found: {train_file}")
        print("Run prepare_dataset.py first.")
        return
    
    print(f"Loading training data from {train_file}...")
    texts = train_file.read_text(encoding='utf-8').strip().split('\n')
    print(f"Loaded {len(texts)} lines")
    
    if tokenizer_type == 'byte_level':
        print("Creating byte-level tokenizer (no training needed)...")
        tokenizer = ByteLevelTokenizer()
    else:
        print("Training BPE tokenizer...")
        tokenizer = BPETokenizer(config['tokenizer']['vocab_size'])
        tokenizer.train(texts)
    
    tokenizer.save(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
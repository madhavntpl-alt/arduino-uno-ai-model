#!/usr/bin/env python3
"""
prepare_dataset.py — Dataset cleaning and preparation for UNO-MIND

Reads raw text files from data/raw/, cleans and normalizes them,
writes cleaned output to data/cleaned/, and creates train/validation split.
"""

import os
import re
import hashlib
import argparse
import yaml
from pathlib import Path
from typing import List, Set, Iterator
import sys

# Add parent directory to path for config loading
sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def is_valid_utf8(text: bytes) -> bool:
    """Check if bytes are valid UTF-8."""
    try:
        text.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def clean_text(text: str, config: dict) -> str:
    """
    Clean and normalize text.
    - Remove control characters (keep \n, \t)
    - Normalize whitespace
    - Filter by length
    """
    # Remove control chars except newline and tab
    text = ''.join(ch for ch in text if ch == '\n' or ch == '\t' or ord(ch) >= 32)
    
    # Normalize whitespace: collapse multiple spaces/tabs, but preserve newlines
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Collapse horizontal whitespace
        line = re.sub(r'[ \t]+', ' ', line)
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    return text


def passes_filters(line: str, config: dict) -> bool:
    """Check if line passes length and content filters."""
    min_len = config['dataset']['min_line_length']
    max_len = config['dataset']['max_line_length']
    return min_len <= len(line) <= max_len


def deduplicate_lines(lines: List[str]) -> List[str]:
    """Remove exact duplicate lines using hash set."""
    seen: Set[str] = set()
    unique = []
    for line in lines:
        h = hashlib.sha256(line.encode('utf-8')).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(line)
    return unique


def process_file(input_path: Path, output_path: Path, config: dict) -> int:
    """Process a single raw file -> cleaned output. Returns line count."""
    print(f"Processing: {input_path.name}")
    
    try:
        raw_bytes = input_path.read_bytes()
    except Exception as e:
        print(f"  Error reading: {e}")
        return 0
    
    if not is_valid_utf8(raw_bytes):
        print(f"  Warning: Invalid UTF-8, attempting recovery")
        text = raw_bytes.decode('utf-8', errors='ignore')
    else:
        text = raw_bytes.decode('utf-8')
    
    # Clean
    text = clean_text(text, config)
    
    # Split into lines and filter
    lines = text.split('\n')
    lines = [ln for ln in lines if passes_filters(ln, config)]
    
    # Deduplicate if enabled
    if config['dataset']['deduplicate']:
        before = len(lines)
        lines = deduplicate_lines(lines)
        print(f"  Deduplicated: {before} -> {len(lines)} lines")
    
    # Write
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"  Written: {len(lines)} lines -> {output_path}")
    return len(lines)


def create_train_val_split(cleaned_dir: Path, train_path: Path, val_path: Path, val_split: float, seed: int):
    """Combine all cleaned files and split into train/validation."""
    import random
    random.seed(seed)
    
    all_lines = []
    for file_path in sorted(cleaned_dir.glob("*.txt")):
        lines = file_path.read_text(encoding='utf-8').strip().split('\n')
        all_lines.extend([ln for ln in lines if ln.strip()])
    
    print(f"Total cleaned lines: {len(all_lines)}")
    
    # Shuffle
    random.shuffle(all_lines)
    
    # Split
    val_size = int(len(all_lines) * val_split)
    val_lines = all_lines[:val_size]
    train_lines = all_lines[val_size:]
    
    train_path.write_text('\n'.join(train_lines) + '\n', encoding='utf-8')
    val_path.write_text('\n'.join(val_lines) + '\n', encoding='utf-8')
    
    print(f"Train: {len(train_lines)} lines -> {train_path}")
    print(f"Val:   {len(val_lines)} lines -> {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for UNO-MIND")
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--raw-dir', help='Override raw data directory')
    parser.add_argument('--cleaned-dir', help='Override cleaned data directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(config['dataset']['raw_dir'])
    cleaned_dir = Path(args.cleaned_dir) if args.cleaned_dir else Path(config['dataset']['cleaned_dir'])
    train_file = Path(config['dataset']['train_file'])
    val_file = Path(config['dataset']['val_file'])
    
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    train_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process each raw file
    raw_files = list(raw_dir.glob("*.txt")) + list(raw_dir.glob("*.md")) + list(raw_dir.glob("*.json"))
    if not raw_files:
        print(f"No raw files found in {raw_dir}. Add .txt, .md, or .json files.")
        return
    
    total_lines = 0
    for raw_file in raw_files:
        out_file = cleaned_dir / f"{raw_file.stem}_cleaned.txt"
        total_lines += process_file(raw_file, out_file, config)
    
    print(f"\nTotal processed lines: {total_lines}")
    
    # Create train/val split
    create_train_val_split(cleaned_dir, train_file, val_file, 
                          config['dataset']['val_split'], config['training']['seed'])
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
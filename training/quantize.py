#!/usr/bin/env python3
"""
quantize.py — Post-training quantization and QAT model conversion for UNO-MIND

Converts trained float32 model to INT8 quantized model suitable for AVR deployment.
Supports both QAT (preferred) and PTQ (fallback) paths.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import json

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


class QuantizedModel:
    """Container for quantized model weights and scales."""
    
    def __init__(self):
        self.embedding_weight = None      # INT8 [vocab_size, embed_dim]
        self.embedding_scale = None       # float32
        self.hidden_weight = None         # INT8 [hidden_dim, input_dim]
        self.hidden_scale = None          # float32
        self.hidden_bias = None           # INT32 [hidden_dim]
        self.hidden_bias_scale = None     # float32
        self.output_weight = None         # INT8 [vocab_size, hidden_dim]
        self.output_scale = None          # float32
        self.output_bias = None           # INT32 [vocab_size]
        self.output_bias_scale = None     # float32
        self.tie_weights = False
        
        # Metadata
        self.vocab_size = 0
        self.embed_dim = 0
        self.context_window = 0
        self.hidden_dim = 0
        self.input_dim = 0
        
    def save_npz(self, path: Path):
        """Save as compressed NPZ for inspection."""
        data = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                val = getattr(self, attr)
                if val is not None:
                    data[attr] = val
        np.savez_compressed(path, **data)
        print(f"Quantized model saved to {path}")
    
    def load_npz(self, path: Path):
        """Load from NPZ."""
        data = np.load(path)
        for key in data.files:
            setattr(self, key, data[key])
        print(f"Quantized model loaded from {path}")


def quantize_tensor_symmetric(tensor: torch.Tensor, dtype=torch.int8) -> tuple:
    """
    Symmetric per-tensor quantization.
    Returns (quantized_tensor, scale)
    """
    # Handle empty tensors
    if tensor.numel() == 0:
        return torch.zeros_like(tensor, dtype=dtype), 1.0
    
    # Get max absolute value
    max_val = tensor.abs().max().item()
    if max_val == 0:
        return torch.zeros_like(tensor, dtype=dtype), 1.0
    
    # Symmetric range: [-127, 127] for int8 (reserve -128 for safety)
    qmax = 127
    scale = max_val / qmax
    
    # Quantize
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(dtype)
    
    return quantized, scale


def quantize_bias_symmetric(bias: torch.Tensor, weight_scale: float, input_scale: float) -> tuple:
    """
    Quantize bias for INT32 accumulation.
    bias_quantized = round(bias / (weight_scale * input_scale))
    """
    if bias is None or bias.numel() == 0:
        return torch.zeros(1, dtype=torch.int32), 1.0
    
    scale = weight_scale * input_scale
    if scale == 0:
        scale = 1.0
    
    quantized = torch.round(bias / scale).to(torch.int32)
    return quantized, scale


def quantize_model_qat(qat_model: nn.Module, config: dict) -> QuantizedModel:
    """Extract quantized weights from QAT model."""
    qmodel = QuantizedModel()
    mcfg = config['model']
    
    qmodel.vocab_size = mcfg['vocab_size']
    qmodel.embed_dim = mcfg['embed_dim']
    qmodel.context_window = mcfg['context_window']
    qmodel.hidden_dim = mcfg['hidden_dim']
    qmodel.input_dim = mcfg['context_window'] * mcfg['embed_dim']
    qmodel.tie_weights = mcfg['tie_weights']
    
    # QAT model has FakeQuantize modules - get quantized weights
    qat_model.eval()
    
    # Embedding
    if hasattr(qat_model, 'embedding'):
        # Get weight from embedding's fake quantize
        emb_weight = qat_model.embedding.weight.detach()
        qmodel.embedding_weight, qmodel.embedding_scale = quantize_tensor_symmetric(emb_weight)
        print(f"Embedding: scale={qmodel.embedding_scale:.6f}, "
              f"range=[{qmodel.embedding_weight.min()}, {qmodel.embedding_weight.max()}]")
    
    # Hidden layer
    if hasattr(qat_model, 'hidden'):
        hid_weight = qat_model.hidden.weight.detach()
        qmodel.hidden_weight, qmodel.hidden_scale = quantize_tensor_symmetric(hid_weight)
        
        hid_bias = qat_model.hidden.bias.detach()
        qmodel.hidden_bias, qmodel.hidden_bias_scale = quantize_bias_symmetric(
            hid_bias, qmodel.hidden_scale, qmodel.embedding_scale)
        print(f"Hidden: weight_scale={qmodel.hidden_scale:.6f}, "
              f"bias_scale={qmodel.hidden_bias_scale:.6f}")
    
    # Output layer
    if hasattr(qat_model, 'output'):
        out_weight = qat_model.output.weight.detach()
        qmodel.output_weight, qmodel.output_scale = quantize_tensor_symmetric(out_weight)
        
        out_bias = qat_model.output.bias.detach()
        qmodel.output_bias, qmodel.output_bias_scale = quantize_bias_symmetric(
            out_bias, qmodel.output_scale, qmodel.hidden_scale)
        print(f"Output: weight_scale={qmodel.output_scale:.6f}, "
              f"bias_scale={qmodel.output_bias_scale:.6f}")
    
    return qmodel


def quantize_model_ptq(model: nn.Module, config: dict, calib_loader) -> QuantizedModel:
    """Post-training quantization using calibration data."""
    print("Running Post-Training Quantization (PTQ)...")
    
    # For PTQ, we use PyTorch's static quantization
    model.eval()
    
    # Prepare for static quantization
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibration
    print(f"Calibrating on {len(calib_loader)} batches...")
    with torch.no_grad():
        for i, (contexts, _) in enumerate(calib_loader):
            if i >= config['quantization']['calibration_samples'] // config['training']['batch_size']:
                break
            model(contexts)
    
    # Convert
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    # Extract quantized weights
    qmodel = QuantizedModel()
    mcfg = config['model']
    qmodel.vocab_size = mcfg['vocab_size']
    qmodel.embed_dim = mcfg['embed_dim']
    qmodel.context_window = mcfg['context_window']
    qmodel.hidden_dim = mcfg['hidden_dim']
    qmodel.input_dim = mcfg['context_window'] * mcfg['embed_dim']
    qmodel.tie_weights = mcfg['tie_weights']
    
    # Extract from quantized model (this is complex with PyTorch's quantized modules)
    # For simplicity, fall back to symmetric quantization of float weights
    print("Warning: PTQ extraction simplified - using symmetric quantization of float weights")
    
    with torch.no_grad():
        # Embedding
        emb_weight = model.embedding.weight
        qmodel.embedding_weight, qmodel.embedding_scale = quantize_tensor_symmetric(emb_weight)
        
        # Hidden
        hid_weight = model.hidden.weight
        qmodel.hidden_weight, qmodel.hidden_scale = quantize_tensor_symmetric(hid_weight)
        hid_bias = model.hidden.bias
        qmodel.hidden_bias, qmodel.hidden_bias_scale = quantize_bias_symmetric(
            hid_bias, qmodel.hidden_scale, qmodel.embedding_scale)
        
        # Output
        out_weight = model.output.weight
        qmodel.output_weight, qmodel.output_scale = quantize_tensor_symmetric(out_weight)
        out_bias = model.output.bias
        qmodel.output_bias, qmodel.output_bias_scale = quantize_bias_symmetric(
            out_bias, qmodel.output_scale, qmodel.hidden_scale)
    
    return qmodel


def validate_quantization(float_model: nn.Module, qmodel: QuantizedModel, 
                          val_loader, device: torch.device, config: dict) -> dict:
    """Compare float vs quantized model outputs."""
    print("\nValidating quantization...")
    
    float_model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_float_loss = 0.0
    total_quant_loss = 0.0
    total_tokens = 0
    max_logit_diff = 0.0
    
    with torch.no_grad():
        for contexts, targets in val_loader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            batch_size = contexts.size(0)
            
            # Float model
            float_logits = float_model(contexts)
            float_loss = criterion(float_logits, targets)
            total_float_loss += float_loss.item() * batch_size
            
            # Quantized inference (simulated in float)
            quant_logits = quantized_forward(qmodel, contexts, config)
            quant_loss = criterion(quant_logits, targets)
            total_quant_loss += quant_loss.item() * batch_size
            
            # Max logit difference
            diff = (float_logits - quant_logits).abs().max().item()
            max_logit_diff = max(max_logit_diff, diff)
            
            total_tokens += batch_size
            
            if total_tokens >= 1000:  # Quick validation
                break
    
    avg_float_loss = total_float_loss / total_tokens
    avg_quant_loss = total_quant_loss / total_tokens
    
    results = {
        'float_loss': avg_float_loss,
        'quant_loss': avg_quant_loss,
        'loss_degradation': avg_quant_loss - avg_float_loss,
        'relative_degradation': (avg_quant_loss - avg_float_loss) / avg_float_loss,
        'max_logit_diff': max_logit_diff,
    }
    
    print(f"Float loss: {avg_float_loss:.4f}")
    print(f"Quant loss: {avg_quant_loss:.4f}")
    print(f"Degradation: {results['loss_degradation']:.4f} ({results['relative_degradation']*100:.1f}%)")
    print(f"Max logit diff: {max_logit_diff:.4f}")
    
    return results


def quantized_forward(qmodel: QuantizedModel, contexts: torch.Tensor, config: dict) -> torch.Tensor:
    """Simulate quantized forward pass in float for validation."""
    device = contexts.device
    batch_size = contexts.size(0)
    
    # Dequantize embeddings
    emb_weight = qmodel.embedding_weight.to(device).float() * qmodel.embedding_scale
    x = F.embedding(contexts, emb_weight)  # [batch, ctx, embed_dim]
    x = x.view(batch_size, -1)  # [batch, input_dim]
    
    # Hidden layer
    hid_weight = qmodel.hidden_weight.to(device).float() * qmodel.hidden_scale
    hid_bias = qmodel.hidden_bias.to(device).float() * qmodel.hidden_bias_scale
    x = F.linear(x, hid_weight, hid_bias)
    x = F.relu(x)
    
    # Output layer
    if qmodel.tie_weights:
        out_weight = emb_weight.t()  # [vocab_size, embed_dim]
        # Need projection if hidden_dim != embed_dim
        if qmodel.hidden_dim != qmodel.embed_dim:
            # This would need a projection layer - simplified
            pass
    else:
        out_weight = qmodel.output_weight.to(device).float() * qmodel.output_scale
    out_bias = qmodel.output_bias.to(device).float() * qmodel.output_bias_scale
    
    logits = F.linear(x, out_weight, out_bias)
    return logits


def main():
    parser = argparse.ArgumentParser(description="Quantize UNO-MIND model")
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--model', help='Input model checkpoint (.pt)')
    parser.add_argument('--output', help='Output quantized model (.npz)')
    parser.add_argument('--val-file', help='Validation data for calibration/validation')
    parser.add_argument('--method', choices=['qat', 'ptq'], default='qat', help='Quantization method')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cpu')  # Quantization on CPU
    
    model_path = Path(args.model) if args.model else Path("uno_mind_model.pt")
    output_path = Path(args.output) if args.output else Path("uno_mind_quantized.npz")
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    
    # Load model
    from train_model import UnoMindModel, TextDataset, get_device
    
    checkpoint = torch.load(model_path, map_location=device)
    mcfg = config['model']
    
    model = UnoMindModel(
        vocab_size=mcfg['vocab_size'],
        embed_dim=mcfg['embed_dim'],
        context_window=mcfg['context_window'],
        hidden_dim=mcfg['hidden_dim'],
        tie_weights=mcfg['tie_weights']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')})")
    
    # Quantize
    if args.method == 'qat' and config['training']['qat']['enabled']:
        print("Using Quantization-Aware Training weights...")
        # Prepare QAT model and load state
        qat_model = torch.quantization.prepare_qat(model, inplace=False)
        qat_model.load_state_dict(checkpoint['model_state_dict'])
        qmodel = quantize_model_qat(qat_model, config)
    else:
        print("Using Post-Training Quantization...")
        # Create calibration loader
        val_file = Path(args.val_file) if args.val_file else Path(config['dataset']['val_file'])
        bos_id = config['tokenizer']['special_tokens']['bos']
        eos_id = config['tokenizer']['special_tokens']['eos']
        calib_dataset = TextDataset(val_file, mcfg['context_window'], bos_id, eos_id, stride=16)
        calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        qmodel = quantize_model_ptq(model, config, calib_loader)
    
    # Validate
    val_file = Path(args.val_file) if args.val_file else Path(config['dataset']['val_file'])
    bos_id = config['tokenizer']['special_tokens']['bos']
    eos_id = config['tokenizer']['special_tokens']['eos']
    val_dataset = TextDataset(val_file, mcfg['context_window'], bos_id, eos_id, stride=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    validate_quantization(model, qmodel, val_loader, device, config)
    
    # Save
    qmodel.save_npz(output_path)
    print(f"\nQuantization complete! Output: {output_path}")


if __name__ == "__main__":
    main()
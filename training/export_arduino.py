#!/usr/bin/env python3
"""
export_arduino.py — Export quantized model to Arduino C++ header

Generates model_data.h with PROGMEM arrays for all weights, scales, and metadata.
"""

import argparse
import yaml
import numpy as np
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_array_c(arr: np.ndarray, name: str, indent: str = "    ") -> str:
    """Format numpy array as C array initializer."""
    if arr.ndim == 1:
        # 1D array
        elements = ', '.join(str(int(x)) for x in arr.flatten())
        return f"{indent}{{{elements}}}"
    elif arr.ndim == 2:
        # 2D array - row major
        rows = []
        for row in arr:
            elements = ', '.join(str(int(x)) for x in row)
            rows.append(f"{indent}    {{{elements}}}")
        return "{\n" + ",\n".join(rows) + f"\n{indent}}}"
    else:
        raise ValueError(f"Unsupported array dimension: {arr.ndim}")


def format_float_array_c(arr: np.ndarray, name: str, indent: str = "    ") -> str:
    """Format float array as C array."""
    if arr.ndim == 0:
        # Scalar
        return f"{arr.item():.8f}f"
    elif arr.ndim == 1:
        elements = ', '.join(f"{x:.8f}f" for x in arr.flatten())
        return f"{indent}{{{elements}}}"
    else:
        raise ValueError(f"Unsupported float array dimension: {arr.ndim}")


def compute_checksum(data: bytes) -> str:
    """Compute SHA256 checksum."""
    return hashlib.sha256(data).hexdigest()[:16]


def generate_model_header(qmodel, config: dict, export_config: dict) -> str:
    """Generate the complete model_data.h content."""
    
    mcfg = config['model']
    tcfg = config['training']
    qcfg = config['quantization']
    
    # Prepare metadata
    metadata = {
        'model_name': export_config['model_name'],
        'model_version': export_config['model_version'],
        'tokenizer_version': export_config['tokenizer_version'],
        'quantization': export_config['quantization'],
        'vocab_size': int(qmodel.vocab_size),
        'embed_dim': int(qmodel.embed_dim),
        'context_window': int(qmodel.context_window),
        'hidden_dim': int(qmodel.hidden_dim),
        'input_dim': int(qmodel.input_dim),
        'tie_weights': bool(qmodel.tie_weights),
        'parameter_count': int(
            qmodel.vocab_size * qmodel.embed_dim +  # embeddings
            qmodel.hidden_dim * qmodel.input_dim +  # hidden weights
            qmodel.hidden_dim +                      # hidden bias
            qmodel.vocab_size * qmodel.hidden_dim +  # output weights
            qmodel.vocab_size                        # output bias
        ),
        'export_timestamp': datetime.utcnow().isoformat() + 'Z',
    }
    
    # Compute checksums for each tensor
    checksums = {}
    
    # Convert to numpy if needed
    def to_numpy(tensor):
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()
        elif hasattr(tensor, 'detach'):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)
    
    emb_w = to_numpy(qmodel.embedding_weight).astype(np.int8)
    hid_w = to_numpy(qmodel.hidden_weight).astype(np.int8)
    hid_b = to_numpy(qmodel.hidden_bias).astype(np.int32)
    out_w = to_numpy(qmodel.output_weight).astype(np.int8)
    out_b = to_numpy(qmodel.output_bias).astype(np.int32)
    
    checksums['embedding_weight'] = compute_checksum(emb_w.tobytes())
    checksums['hidden_weight'] = compute_checksum(hid_w.tobytes())
    checksums['hidden_bias'] = compute_checksum(hid_b.tobytes())
    checksums['output_weight'] = compute_checksum(out_w.tobytes())
    checksums['output_bias'] = compute_checksum(out_b.tobytes())
    
    # Scale values
    emb_scale = float(qmodel.embedding_scale)
    hid_scale = float(qmodel.hidden_scale)
    hid_bias_scale = float(qmodel.hidden_bias_scale)
    out_scale = float(qmodel.output_scale)
    out_bias_scale = float(qmodel.output_bias_scale)
    
    # Build header
    lines = []
    lines.append("// ============================================================")
    lines.append(f"// UNO-MIND Model Data - Auto-generated")
    lines.append(f "// Model: {metadata['model_name']} v{metadata['model_version']}")
    lines.append(f "// Exported: {metadata['export_timestamp']}")
    lines.append("// DO NOT EDIT MANUALLY")
    lines.append("// ============================================================")
    lines.append("")
    lines.append("#ifndef MODEL_DATA_H")
    lines.append("#define MODEL_DATA_H")
    lines.append("")
    lines.append("#include <avr/pgmspace.h>")
    lines.append("#include <stdint.h>")
    lines.append("")
    
    # Metadata constants
    lines.append("// --- Model Metadata ---")
    lines.append(f"#define MODEL_NAME \"{metadata['model_name']}\"")
    lines.append(f"#define MODEL_VERSION \"{metadata['model_version']}\"")
    lines.append(f"#define TOKENIZER_VERSION \"{metadata['tokenizer_version']}\"")
    lines.append(f"#define QUANTIZATION \"{metadata['quantization']}\"")
    lines.append("")
    lines.append(f"#define VOCAB_SIZE {metadata['vocab_size']}")
    lines.append(f"#define EMBED_DIM {metadata['embed_dim']}")
    lines.append(f"#define CONTEXT_WINDOW {metadata['context_window']}")
    lines.append(f"#define HIDDEN_DIM {metadata['hidden_dim']}")
    lines.append(f"#define INPUT_DIM {metadata['input_dim']}")
    lines.append(f"#define TIE_WEIGHTS {1 if metadata['tie_weights'] else 0}")
    lines.append(f"#define PARAMETER_COUNT {metadata['parameter_count']}")
    lines.append("")
    
    # Quantization scales (stored as float for dequantization)
    lines.append("// --- Quantization Scales ---")
    lines.append(f"const float EMBEDDING_SCALE PROGMEM = {emb_scale:.8f}f;")
    lines.append(f"const float HIDDEN_WEIGHT_SCALE PROGMEM = {hid_scale:.8f}f;")
    lines.append(f"const float HIDDEN_BIAS_SCALE PROGMEM = {hid_bias_scale:.8f}f;")
    lines.append(f"const float OUTPUT_WEIGHT_SCALE PROGMEM = {out_scale:.8f}f;")
    lines.append(f"const float OUTPUT_BIAS_SCALE PROGMEM = {out_bias_scale:.8f}f;")
    lines.append("")
    
    # Special tokens
    tok_cfg = config['tokenizer']['special_tokens']
    lines.append("// --- Special Tokens ---")
    lines.append(f"#define TOKEN_BOS {tok_cfg['bos']}")
    lines.append(f"#define TOKEN_EOS {tok_cfg['eos']}")
    lines.append("")
    
    # Embedding table [VOCAB_SIZE][EMBED_DIM]
    lines.append("// --- Embedding Table (INT8) ---")
    lines.append(f"const int8_t EMBEDDING_TABLE[{metadata['vocab_size']}][{metadata['embed_dim']}] PROGMEM = ")
    lines.append(format_array_c(emb_w, "EMBEDDING_TABLE") + ";")
    lines.append("")
    
    # Hidden weights [HIDDEN_DIM][INPUT_DIM]
    lines.append("// --- Hidden Layer Weights (INT8) ---")
    lines.append(f"const int8_t HIDDEN_WEIGHTS[{metadata['hidden_dim']}][{metadata['input_dim']}] PROGMEM = ")
    lines.append(format_array_c(hid_w, "HIDDEN_WEIGHTS") + ";")
    lines.append("")
    
    # Hidden bias [HIDDEN_DIM] (INT32)
    lines.append("// --- Hidden Layer Bias (INT32) ---")
    lines.append(f"const int32_t HIDDEN_BIAS[{metadata['hidden_dim']}] PROGMEM = ")
    lines.append(format_array_c(hid_b, "HIDDEN_BIAS") + ";")
    lines.append("")
    
    # Output weights [VOCAB_SIZE][HIDDEN_DIM]
    if not metadata['tie_weights']:
        lines.append("// --- Output Layer Weights (INT8) ---")
        lines.append(f"const int8_t OUTPUT_WEIGHTS[{metadata['vocab_size']}][{metadata['hidden_dim']}] PROGMEM = ")
        lines.append(format_array_c(out_w, "OUTPUT_WEIGHTS") + ";")
        lines.append("")
    else:
        lines.append("// --- Output Layer: TIED to Embedding (transposed) ---")
        lines.append("// OUTPUT_WEIGHTS = EMBEDDING_TABLE^T (not stored separately)")
        lines.append("")
    
    # Output bias [VOCAB_SIZE] (INT32)
    lines.append("// --- Output Layer Bias (INT32) ---")
    lines.append(f"const int32_t OUTPUT_BIAS[{metadata['vocab_size']}] PROGMEM = ")
    lines.append(format_array_c(out_b, "OUTPUT_BIAS") + ";")
    lines.append("")
    
    # Checksums
    if export_config['include_checksum']:
        lines.append("// --- Checksums (for validation) ---")
        for name, checksum in checksums.items():
            lines.append(f"#define CHECKSUM_{name.upper()} 0x{checksum}")
        lines.append("")
    
    # Model info struct for /model command
    if export_config['include_metadata']:
        lines.append("// --- Model Info (for runtime queries) ---")
        lines.append("struct ModelInfo {")
        lines.append("    const char* name;")
        lines.append("    const char* version;")
        lines.append("    const char* tokenizer;")
        lines.append("    const char* quantization;")
        lines.append("    uint16_t vocab_size;")
        lines.append("    uint16_t embed_dim;")
        lines.append("    uint16_t context_window;")
        lines.append("    uint16_t hidden_dim;")
        lines.append("    uint32_t param_count;")
        lines.append("    uint16_t flash_model_bytes;")
        lines.append("    uint16_t sram_runtime_bytes;")
        lines.append("    uint16_t max_context_tokens;")
        lines.append("    uint16_t max_output_tokens;")
        lines.append("};")
        lines.append("")
        lines.append("const ModelInfo MODEL_INFO PROGMEM = {")
        lines.append(f'    MODEL_NAME,')
        lines.append(f'    MODEL_VERSION,')
        lines.append(f'    TOKENIZER_VERSION,')
        lines.append(f'    QUANTIZATION,')
        lines.append(f'    VOCAB_SIZE,')
        lines.append(f'    EMBED_DIM,')
        lines.append(f'    CONTEXT_WINDOW,')
        lines.append(f'    HIDDEN_DIM,')
        lines.append(f'    PARAMETER_COUNT,')
        lines.append(f'    {estimate_flash_usage(metadata)},  // flash_model_bytes')
        lines.append(f'    {estimate_sram_usage(metadata)},  // sram_runtime_bytes')
        lines.append(f'    CONTEXT_WINDOW,')
        lines.append(f'    128,  // max_output_tokens')
        lines.append("};")
        lines.append("")
    
    lines.append("#endif // MODEL_DATA_H")
    lines.append("")
    
    return '\n'.join(lines)


def estimate_flash_usage(metadata: dict) -> int:
    """Estimate Flash usage for model weights in bytes."""
    vocab = metadata['vocab_size']
    embed = metadata['embed_dim']
    hidden = metadata['hidden_dim']
    input_dim = metadata['input_dim']
    tie = metadata['tie_weights']
    
    # Embedding table: vocab * embed (int8)
    flash = vocab * embed
    # Hidden weights: hidden * input_dim (int8)
    flash += hidden * input_dim
    # Hidden bias: hidden (int32)
    flash += hidden * 4
    # Output weights: vocab * hidden (int8) - unless tied
    if not tie:
        flash += vocab * hidden
    # Output bias: vocab * int32
    flash += vocab * 4
    # Scales: ~5 floats * 4
    flash += 20
    # Metadata struct: ~50 bytes
    flash += 50
    
    return flash


def estimate_sram_usage(metadata: dict) -> int:
    """Estimate SRAM usage at runtime in bytes."""
    vocab = metadata['vocab_size']
    embed = metadata['embed_dim']
    hidden = metadata['hidden_dim']
    ctx = metadata['context_window']
    input_dim = metadata['input_dim']
    
    # Input buffer
    sram = 128
    # Token context buffer
    sram += ctx
    # Embedding lookup buffer: ctx * embed * int16
    sram += ctx * embed * 2
    # Flattened input: input_dim * int16
    sram += input_dim * 2
    # Hidden activations: hidden * int32 (accumulator) + hidden * int8 (post-ReLU)
    sram += hidden * 4 + hidden
    # Output logits: vocab * int32
    sram += vocab * 4
    # Temp arithmetic
    sram += 200
    # Serial/stack reserve
    sram += 500
    
    return sram


def main():
    parser = argparse.ArgumentParser(description="Export quantized model to Arduino header")
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--quantized', help='Input quantized model (.npz)')
    parser.add_argument('--output', help='Output header file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    export_config = config['export']
    
    quantized_path = Path(args.quantized) if args.quantized else Path("uno_mind_quantized.npz")
    output_path = Path(args.output) if args.output else Path(export_config['output_header'])
    
    if not quantized_path.exists():
        print(f"Quantized model not found: {quantized_path}")
        print("Run quantize.py first.")
        return
    
    # Load quantized model
    data = np.load(quantized_path, allow_pickle=True)
    
    class QModel:
        pass
    qmodel = QModel()
    for key in data.files:
        setattr(qmodel, key, data[key])
    
    print(f"Loaded quantized model from {quantized_path}")
    print(f"Vocab: {qmodel.vocab_size}, Embed: {qmodel.embed_dim}, "
          f"Ctx: {qmodel.context_window}, Hidden: {qmodel.hidden_dim}")
    print(f"Embedding scale: {qmodel.embedding_scale:.6f}")
    print(f"Hidden weight scale: {qmodel.hidden_scale:.6f}")
    print(f"Output weight scale: {qmodel.output_scale:.6f}")
    
    # Generate header
    header_content = generate_model_header(qmodel, config, export_config)
    
    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header_content)
    
    # Print summary
    flash_bytes = estimate_flash_usage({
        'vocab_size': qmodel.vocab_size,
        'embed_dim': qmodel.embed_dim,
        'hidden_dim': qmodel.hidden_dim,
        'input_dim': qmodel.input_dim,
        'tie_weights': qmodel.tie_weights,
    })
    sram_bytes = estimate_sram_usage({
        'vocab_size': qmodel.vocab_size,
        'embed_dim': qmodel.embed_dim,
        'hidden_dim': qmodel.hidden_dim,
        'context_window': qmodel.context_window,
    })
    
    print(f"\nExported to {output_path}")
    print(f"Estimated Flash usage: {flash_bytes:,} bytes ({flash_bytes/1024:.1f} KB)")
    print(f"Estimated SRAM usage: {sram_bytes:,} bytes ({sram_bytes/1024:.1f} KB)")
    print(f"Available Flash: ~28 KB (after bootloader + firmware)")
    print(f"Available SRAM: ~1500 bytes (with safety margin)")
    
    if flash_bytes > 28000:
        print("WARNING: Model may exceed Flash budget!")
    if sram_bytes > 1700:
        print("WARNING: Model may exceed SRAM budget!")
    
    print("\nNext steps:")
    print("1. Copy model_data.h to firmware/ (already done)")
    print("2. Compile and upload uno_mind.ino to Arduino Uno")
    print("3. Open Serial Monitor at 115200 baud")


if __name__ == "__main__":
    main()
# UNO-MIND — Offline Local Language Model for Arduino Uno

A from-scratch miniature language model that runs inference directly on an Arduino Uno (ATmega328P). No internet, no APIs, no cloud — completely self-contained.

## Quick Start

### 1. Hardware Required
- Arduino Uno (ATmega328P)
- USB cable

### 2. Flash Firmware (Using Stub Weights)
```bash
# Open in Arduino IDE
arduino uno_mind.ino

# Or use Arduino CLI
arduino-cli compile --fqbn arduino:avr:uno firmware/
arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno firmware/
```

### 3. Open Serial Monitor
- Baud rate: **115200**
- Line ending: **Newline** (or Both NL & CR)

### 4. Try It
```
YOU > what is arduino
AI > [generates response with stub weights - will be gibberish]
```

**Note:** The included `model_data.h` contains **stub weights**. The model will output gibberish until you train and export real weights (see Training Pipeline below).

---

## Training Pipeline (To Get Real Weights)

### Prerequisites
```bash
# Python 3.8+
pip install torch pyyaml numpy tqdm
# For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 1: Prepare Dataset
```bash
cd training
python prepare_dataset.py
```
- Reads from `../data/raw/*.txt`
- Writes cleaned data to `../data/cleaned/`
- Creates `train.txt` and `validation.txt`

### Step 2: Train Tokenizer (Byte-Level = No Training Needed)
```bash
python train_tokenizer.py
```
- Creates `tokenizer.json` (byte-level config)
- Byte-level tokenizer has fixed 256 vocab, no training required

### Step 3: Train Model
```bash
python train_model.py
```
- Trains compact MLP language model
- Uses Quantization-Aware Training (QAT) by default
- Saves best model to `uno_mind_model.pt`
- Typical: 20-30 epochs, ~5-15 min on CPU, ~1-2 min on GPU

### Step 4: Quantize
```bash
python quantize.py
```
- Converts QAT model to INT8 quantized weights
- Saves `uno_mind_quantized.npz`
- Validates quantization degradation

### Step 5: Export to Arduino
```bash
python export_arduino.py
```
- Generates `../firmware/model_data.h` with PROGMEM arrays
- **Overwrites the stub weights!**
- Reports Flash/SRAM estimates

### Step 6: Re-flash Arduino
```bash
# Recompile and upload with new model_data.h
arduino-cli compile --fqbn arduino:avr:uno ../firmware
arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno ../firmware
```

---

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/stats` | Model statistics |
| `/model` | Detailed model info |
| `/memory` | SRAM/Flash usage |
| `/reset` | Reset conversation |
| `/clear` | Clear context window |

Normal text is treated as a prompt.

---

## Architecture

### Model
- **Type**: Fixed-context MLP (not Transformer, not RNN)
- **Tokenizer**: Byte-level (256 vocab, uint8_t tokens)
- **Context**: 8 tokens (128 bytes input)
- **Embedding**: 256 × 16 = 4,096 params
- **Hidden**: 128 → 32 (ReLU) = 4,096 params  
- **Output**: 32 → 256 = 8,192 params
- **Total**: ~16,384 parameters (INT8 = ~16 KB Flash)

### Quantization
- **Weights**: INT8 symmetric per-tensor
- **Activations**: INT8 (clipped ReLU)
- **Accumulators**: INT32
- **Biases**: INT32 (quantized with combined scale)

### Memory Budget (ATmega328P)
| Region | Budget | Used (est) |
|--------|--------|------------|
| Flash (model) | 28 KB | 16 KB |
| Flash (code) | - | ~8 KB |
| SRAM (runtime) | 1.7 KB | 1.5 KB |
| Stack reserve | - | ~300 B |

---

## Project Structure

```
UNO-MIND/
├── firmware/
│   ├── uno_mind.ino      # Single-file firmware (this is it!)
│   └── model_data.h      # Generated weights (stub included)
├── training/
│   ├── config.yaml       # All hyperparameters
│   ├── prepare_dataset.py
│   ├── train_tokenizer.py
│   ├── train_model.py
│   ├── quantize.py
│   ├── export_arduino.py
│   └── evaluate.py
├── data/
│   ├── raw/              # Add your .txt files here
│   └── cleaned/          # Auto-generated
└── docs/
    └── PRD.md            # Full requirements
```

---

## Adding Training Data

Add `.txt`, `.md`, or `.json` files to `data/raw/`:
```
data/raw/
  ├── my_conversations.txt
  ├── technical_docs.txt
  └── wikipedia_subset.txt
```

Then re-run:
```bash
python prepare_dataset.py
python train_model.py
python quantize.py
python export_arduino.py
```

---

## Expected Performance (After Training)

| Metric | Target |
|--------|--------|
| Validation Perplexity | < 50 |
| Next-token Accuracy | > 25% |
| Generation Speed | 2-5 tokens/sec |
| Coherent Responses | Most common queries |
| Unseen Word Handling | Byte decomposition works |

---

## Troubleshooting

### "model_data.h not found"
- Firmware compiles with defaults (warning shown)
- Run `export_arduino.py` to generate real weights

### Gibberish Output
- **Expected with stub weights!** Train the model first.
- After training: check perplexity < 50, retrain if higher

### Upload Fails
- Check port: `arduino-cli board list`
- Hold reset button during upload if needed
- Try lower baud: change `SERIAL_BAUD` to 9600

### SRAM Overflow
- Reduce `CONTEXT_WINDOW` in config.yaml (and model_data.h)
- Reduce `HIDDEN_DIM` or `EMBED_DIM`
- Check `/memory` command output

### Flash Overflow
- Enable `tie_weights: true` in config.yaml (saves 8 KB)
- Reduce `VOCAB_SIZE` (requires tokenizer change)
- Use INT4 quantization (advanced)

---

## Customization

### Change Model Size
Edit `training/config.yaml`:
```yaml
model:
  embed_dim: 16      # Try 8, 16, 32
  context_window: 8  # Try 4, 8, 16
  hidden_dim: 32     # Try 16, 32, 64
  tie_weights: false # true saves Flash
```

Then retrain and re-export.

### Change Generation
In `uno_mind.ino`:
```cpp
#define MAX_OUTPUT_TOKENS 128  // Response length limit
#define REPETITION_PENALTY 15  // Higher = less repetition
#define REPETITION_WINDOW 8    // Tokens to track
```

---

## Limitations

- **Tiny model**: ~16K params vs billions in modern LLMs
- **Byte-level tokens**: Longer sequences, less semantic compression
- **Fixed context**: Only 8 tokens of history
- **No attention**: Can't model long-range dependencies well
- **Greedy only**: No temperature/top-k sampling (memory constrained)

This is a **demonstration** of the complete pipeline on extreme hardware.

---

## License

MIT — Use freely for learning and experimentation.

---

## Credits

Built as an educational project demonstrating:
- End-to-end ML pipeline (data → train → quantize → deploy)
- Embedded ML on 8-bit microcontrollers
- Quantization-aware training
- PROGMEM and AVR optimization techniques
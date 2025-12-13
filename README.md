# SOREModel - Simple Open-Source Recurrent/Transformer Model

**Version:** 2.1.0  
**Author:** SOREModel Team

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview
SOREModel is an experimental GPT-style transformer implementation designed for simplicity and educational purposes. It features modern architectural choices like Rotary Positional Embeddings (RoPE) and ALiBi (Attention with Linear Biases), enabling efficient and effective text generation.

This repository contains the full source code for training and running inference with SOREModel v3.

## Features

| Feature | SOREModel v1 | SOREModel v2 | SOREModel v3 |
|----------------|--------------|--------------|--------------|
| Architecture | Linear simples | Transformer | Transformer (GPT-style) |
| Multi-Head Attention | No | Yes | Yes (with ALiBi) |
| Position Embedding | No | Yes | Rotary (RoPE) |
| Layer Normalization | No | Yes | RMSNorm / Pre-LN |
| Residual Connections | No | Yes | Yes |
| Complexity | Low | High | High (Optimized) |
| Generation Quality | Basic | Advanced | Experimental SOTA |

## Project Structure

- `src/`: Core source code.
    - `models/`: Model architectures (SOREModel v3).
    - `training/`: Training loop and logic.
    - `data/`: Dataset and tokenization utilities.
    - `generation/`: Text generation pipelines.
- `scripts/`: Entry points for usage.
    - `train.py`: Script to train or fine-tune the model.
    - `chat.py`: Interactive script to chat with a trained model.
- `data/`: Directory for datasets and tokenizer files.

```bash
SOREModel/
    ├── data/               # Data and tokenizer files
    ├── scripts/            # Entry points (train.py, chat.py)
    ├── src/                # Source code
    │   ├── data/           # Dataset classes
    │   ├── generation/     # Text generation logic
    │   ├── models/         # SOREModel_v3 architecture
    │   ├── training/       # Trainer class
    │   └── __init__.py     # Exports
    ├── requirements.txt
    └── README.md
```

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To start training a new model (or resume from checkpoint):

```bash
python scripts/train.py \
    --dataset_name wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --tokenizer_name gpt2 \
    --epochs 10 \
    --batch_size 8 \
    --context_size 512
```

Arguments:
- `--output_dir`: Directory to save checkpoints (default: `./checkpoints`).
- `--resume_from_checkpoint`: Path to a checkpoint directory to resume.
- Use `--help` to see all available options.

### Chat / Inference

To interact with a trained model:

```bash
python scripts/chat.py --checkpoint_path checkpoints/final_model/model.pt
```

Arguments:
- `--tokenizer_name`: Must match the tokenizer used during training (default: `gpt2`).

## License
MIT License

## Contact

- **Project**: SOREModel
- **Version**: 2.1.0
- **Author**: SOREModel Team
- **Issues**: [GitHub Issues](https://github.com/Elitinho123456/SOREModel/issues)

## Acknowledgments

- PyTorch team for the excellent library
- Open-source community for inspiration and support
- Project contributors and users

---

# For Contributors

<div align="center">

**Make with ❤️ by the open-source community**

[⭐ Star](https://github.com/Elitinho123456/SOREModel) | [🐛 Report an issue](https://github.com/Elitinho123456/SOREModel/issues) | [💬 Discussions](https://github.com/Elitinho123456/SOREModel/discussions)

</div>
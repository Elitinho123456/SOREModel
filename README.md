# SOREModel - Simple Open-Source Recurrent/Transformer Model

**Version:** 4.0.0 (Reform)  
**Author:** SOREModel Team

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview
SOREModel v4 is a "home-made GPT-style LLM" architecture designed to be efficient and educational. It mimics modern LLM pipelines on a smaller scale, featuring:
- **SOREModel v4**: Architecture with RoPE, ALiBi, RMSNorm, and Weight Tying.
- **Distillation Pipeline**: Transfer knowledge from OpenAI/Gemini to your small model.
- **Optimization**: AMP training, Quantization, and ONNX export.

## Features

| Feature | SOREModel v3 | SOREModel v4 (Current) |
|----------------|--------------|--------------|
| Architecture | GPT-style | GPT-style Optimized |
| Positional Emb | RoPE | RoPE + ALiBi (Extrapolation) |
| Optimization | Basic | Weight Tying, RMSNorm |
| Training | Simple Loop | AMP, Schedulers, Distillation |
| Inference | Python | Python, Quantized, ONNX |

## Project Structure

- `src/`: Core source code.
    - `models/`: `soreModel_v4.py` (Main).
    - `training/`: `Trainer` and `DistillationTrainer`.
    - `core/`: `TeacherClient` (OpenAI/Gemini integrations).
- `scripts/`:
    - `train.py`: Unified training script (Pretrain/SFT).
    - `distill_sore.py`: Distillation from Teacher models.
    - `quantize_sore.py`: Create CPU-optimized models.
    - `export_onnx_sore.py`: Export to ONNX.
- `docs/`:
    - `training.md`: Guide for Pretrain/Distill/SFT.
    - `performance.md`: Optimization guide.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Pretrain (Autoregressive)
```bash
python scripts/train.py --stage pretrain --dataset_name wikitext --use_amp
```

### 2. Distill (from Gemini/OpenAI)
```bash
python scripts/distill_sore.py --teacher_provider gemini --dataset_name my_prompts
```

### 3. Optimization
```bash
python scripts/quantize_sore.py --checkpoint checkpoints/final_model/model.pt
```

For full details, see [docs/training.md](docs/training.md) and [docs/performance.md](docs/performance.md).

## License
MIT License

## Contact
- **Project**: SOREModel
- **Issues**: [GitHub Issues](https://github.com/Elitinho123456/SOREModel/issues)

---
## Acknowledgments

- PyTorch team for the excellent library
- Open-source community for inspiration and support
- Project contributors and users
- [Max - Machine learning and deep learning enthusiast.](https://github.com/maxmelo1?tab=overview&from=2025-11-01&to=2025-11-30) for helping and teaching me this incredible area of programming, motivating and cheering me up throughout the entire development process

---

<div align="center">
**Made with ❤️ by the open-source community**

[⭐ Star](https://github.com/Elitinho123456/SOREModel) | [🐛 Report an issue](https://github.com/Elitinho123456/SOREModel/issues) | [💬 Discussions](https://github.com/Elitinho123456/SOREModel/discussions)

</div>
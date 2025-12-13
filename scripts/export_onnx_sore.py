
"""
export_onnx_sore.py
Exports SOREModel to ONNX format.
"""
import sys
import os
import argparse
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.soreModel_v4 import SOREModel_v4, ModelConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Export SOREModel to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to input .pt checkpoint')
    parser.add_argument('--output', type=str, default='sore_model.onnx', help='Output ONNX path')
    parser.add_argument('--opset', type=int, default=14, help='ONNX Opset version')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading checkpoint {args.checkpoint}...")
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
    except FileNotFoundError:
        print("Checkpoint file not found.")
        sys.exit(1)

    config_dict = checkpoint.get('config', {})
    if config_dict:
        config = ModelConfig(**config_dict)
    else:
        print("Warning: No config found, using defaults.")
        config = ModelConfig()
        
    model = SOREModel_v4(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Dummy input
    dummy_input = torch.randint(0, config.vocab_size, (1, 128))
    
    print(f"Exporting to {args.output}...")
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print("Export complete!")

if __name__ == '__main__':
    main()

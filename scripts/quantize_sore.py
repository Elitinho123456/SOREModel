
"""
quantize_sore.py
Applies dynamic quantization to SOREModel checkpoints.
Reduces model size and improves CPU inference speed.
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
    parser = argparse.ArgumentParser(description='Quantize SOREModel')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to input .pt checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output path (defaults to *_quantized.pt)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        sys.exit(1)
        
    print(f"Loading checkpoint {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Reconstruct config if available, else assume default/generic
    config_dict = checkpoint.get('config', {})
    if config_dict:
        config = ModelConfig(**config_dict)
    else:
        print("Warning: No config found in checkpoint. Using default V4 config.")
        config = ModelConfig()
        
    model = SOREModel_v4(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Applying dynamic quantization (Linear layers)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.checkpoint)
        output_path = f"{base}_quantized{ext}"
        
    print(f"Saving quantized model to {output_path}...")
    torch.save(quantized_model.state_dict(), output_path)
    
    orig_size = os.path.getsize(args.checkpoint) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! Size reduced from {orig_size:.2f} MB to {new_size:.2f} MB")

if __name__ == '__main__':
    main()

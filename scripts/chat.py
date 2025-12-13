"""
chat.py
Interactive chat script for SOREModel v3.
"""
import sys
import os
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Ensure src is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import SOREModel_v3, TextGenerator
from src.models.soreModel_v3 import ModelConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with SOREModel v3')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint (model.pt)')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='HF Tokenizer Name used in training')
    parser.add_argument('--context_size', type=int, default=512, help='Context Size used in training')
    return parser.parse_args()

def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config from checkpoint if available, else use default/arg based (simplified here)
    # Ideally checkpoint has 'config' dict.
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = ModelConfig(**config_dict)
    else:
        print("Warning: No config found in checkpoint. Using default config.")
        config = ModelConfig() 

    model = SOREModel_v3(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading tokenizer {args.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = load_model(args.checkpoint_path, device)
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    generator = TextGenerator(model, tokenizer, device)

    print("\n--- SOREModel Chat ---")
    print("Type '/quit' to exit.")
    
    while True:
        try:
            prompt = input("\nYou: ")
        except EOFError:
            break
            
        if prompt.strip().lower() == '/quit':
            break
            
        if not prompt.strip():
            continue
            
        response = generator.gerar_texto(
            contexto_inicial=prompt,
            max_length=100,
            temperature=0.8,
            top_k=40
        )
        
        # Simple post-processing to remove prompt if generator includes it
        if response.startswith(prompt):
             response = response[len(prompt):]
             
        print(f"AI: {response}")

if __name__ == '__main__':
    main()
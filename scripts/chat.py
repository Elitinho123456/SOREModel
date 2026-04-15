"""
chat.py
Interactive chat script for SOREModel v4.1 with v3 fallback.
"""
import sys
import argparse
from dataclasses import fields
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Ensure src is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import TextGenerator
from src.models.soreModel_v3 import SOREModel_v3, ModelConfig as ModelConfigV3
from src.models.soreModel_v4_1 import SOREModel_v4_1, ModelConfig as ModelConfigV4_1

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with SOREModel')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint (model.pt)')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='HF Tokenizer Name used in training')
    parser.add_argument('--context_size', type=int, default=None, help='Optional context size override')
    parser.add_argument('--model_version', type=str, default='auto', choices=['auto', 'v4_1', 'v3'], help='Model architecture used by checkpoint')
    return parser.parse_args()

def _filter_config(config_dict, config_cls):
    valid_fields = {f.name for f in fields(config_cls)}
    return {k: v for k, v in config_dict.items() if k in valid_fields}

def _resolve_model_classes(config_dict, model_version):
    if model_version == 'v4_1':
        return SOREModel_v4_1, ModelConfigV4_1, 'v4_1'
    if model_version == 'v3':
        return SOREModel_v3, ModelConfigV3, 'v3'

    # Auto detection: v3 checkpoints usually contain rotary_pct in config.
    detected = 'v3' if 'rotary_pct' in config_dict else 'v4_1'
    if detected == 'v3':
        return SOREModel_v3, ModelConfigV3, detected
    return SOREModel_v4_1, ModelConfigV4_1, detected

def load_model(checkpoint_path, device, model_version='auto', context_size=None):
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config_dict = checkpoint.get('config', {})
    model_cls, config_cls, resolved_version = _resolve_model_classes(config_dict, model_version)
    print(f"Using model architecture: {resolved_version}")

    if config_dict:
        filtered_config = _filter_config(config_dict, config_cls)
        config = config_cls(**filtered_config)
    else:
        print('Warning: No config found in checkpoint. Using default config.')
        config = config_cls()

    if context_size is not None:
        config.context_size = context_size

    model = model_cls(config)
    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
    if state_dict is None:
        raise KeyError("Checkpoint must contain 'model_state_dict' or 'state_dict'.")

    model.load_state_dict(state_dict)
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
        model = load_model(
            args.checkpoint_path,
            device,
            model_version=args.model_version,
            context_size=args.context_size,
        )
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
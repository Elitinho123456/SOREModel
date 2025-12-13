"""
train.py
Main training script for SOREModel v3.
"""
import sys
import os
import argparse
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

# Ensure src is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import SOREModel_v3, Trainer, TextDataset
from src.models.soreModel_v3 import ModelConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train SOREModel v3')
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset Name')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1', help='Dataset Config')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output Directory')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='HF Tokenizer Name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--context_size', type=int, default=512, help='Context Size')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup Steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Grad Accumulation')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save Steps')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Checkpoint Path')
    
    # Model Configs
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--no_alibi', action='store_true')
    parser.add_argument('--no_rmsnorm', action='store_true')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    
    texts = []
    for split in ['train', 'validation', 'test']:
        if split in dataset:
            clean_texts = [t for t in dataset[split]['text'] if t.strip()]
            texts.extend(clean_texts)
            
    print(f"Loaded {len(texts)} documents.")
    
    print(f"Loading tokenizer {args.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Creating dataset...")
    train_dataset = TextDataset(texts, tokenizer, args.context_size)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    print("Configuring model...")
    config = ModelConfig(
        vocab_size=len(tokenizer),
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_alibi=(not args.no_alibi),
        use_rmsnorm=(not args.no_rmsnorm)
    )
    
    model = SOREModel_v3(config)
    
    trainer = Trainer(model, tokenizer, args)
    
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
        
    trainer.train(train_loader, args.epochs)
    
    trainer.save_checkpoint('final_model')

if __name__ == '__main__':
    main()

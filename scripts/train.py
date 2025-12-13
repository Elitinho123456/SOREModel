
"""
train.py
Main training script for SOREModel v4 (and v3 legacy).
Supports:
- Pretraining vs SFT stages
- SOREModel v4 (default)
- AMP, Schedulers, Early Stopping
"""
import sys
import os
import argparse
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

# Ensure src is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src import Trainer, TextDataset
from src.data.dataset import InstructionDataset
# Explicitly import v4
from src.models.soreModel_v4 import SOREModel_v4, ModelConfig as ConfigV4
from src.models.soreModel_v3 import SOREModel_v3, ModelConfig as ConfigV3
from config import TRAINING_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description='Train SOREModel')
    
    # Stage
    parser.add_argument('--stage', type=str, default='pretrain', choices=['pretrain', 'sft'], help='Training Stage')
    
    # Data
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset Name')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1', help='Dataset Config')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output Directory')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='HF Tokenizer Name')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Validation Split Ratio')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
    parser.add_argument('--context_size', type=int, default=1024, help='Context Size')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Grad Accumulation')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save Steps')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Checkpoint Path')
    
    # Advanced Training (Defaults from config if not provided)
    parser.add_argument('--use_amp', action='store_true', default=TRAINING_CONFIG['use_amp'])
    parser.add_argument('--lr_scheduler', type=str, default=TRAINING_CONFIG['lr_scheduler'])
    parser.add_argument('--warmup_steps', type=int, default=TRAINING_CONFIG['warmup_steps'])
    parser.add_argument('--early_stopping_patience', type=int, default=TRAINING_CONFIG['early_stopping_patience'])
    parser.add_argument('--min_delta', type=float, default=TRAINING_CONFIG['min_delta'])
    
    # Model Configs
    parser.add_argument('--model_version', type=str, default='v4', choices=['v3', 'v4'])
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--no_alibi', action='store_true')
    parser.add_argument('--no_rmsnorm', action='store_true') # relevant for v3 override, v4 usually forces it or uses config
    
    return parser.parse_args()

def load_data(args, tokenizer):
    print(f"Loading dataset {args.dataset_name}...")
    if args.stage == 'pretrain':
        # Expect generic text dataset
        dataset = load_dataset(args.dataset_name, args.dataset_config)
        texts = []
        # Support common splits
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                clean_texts = [t for t in dataset[split]['text'] if t.strip()]
                texts.extend(clean_texts)
        
        # Simple random split for val
        val_size = int(len(texts) * args.validation_split)
        train_texts = texts[:-val_size] if val_size > 0 else texts
        val_texts = texts[-val_size:] if val_size > 0 else []
        
        print(f"Pretrain: {len(train_texts)} train docs, {len(val_texts)} val docs")
        
        train_ds = TextDataset(train_texts, tokenizer, args.context_size)
        val_ds = TextDataset(val_texts, tokenizer, args.context_size) if val_texts else None
        
    elif args.stage == 'sft':
        # Expect dataset with instruction/output - e.g. "yahma/alpaca-cleaned"
        # Since structure varies, we assume standard HF format or 'text' field if processed
        try:
            dataset = load_dataset(args.dataset_name, args.dataset_config)
        except:
             print("Could not load dataset directly, check name.")
             sys.exit(1)
             
        # Flatten
        all_data = []
        for split in dataset.keys():
            all_data.extend([item for item in dataset[split]])
            
        val_size = int(len(all_data) * args.validation_split)
        train_data = all_data[:-val_size] if val_size > 0 else all_data
        val_data = all_data[-val_size:] if val_size > 0 else []
        
        print(f"SFT: {len(train_data)} train samples, {len(val_data)} val samples")
        
        train_ds = InstructionDataset(train_data, tokenizer, args.context_size)
        val_ds = InstructionDataset(val_data, tokenizer, args.context_size) if val_data else None

    return train_ds, val_ds

def main():
    args = parse_args()
    
    print(f"Loading tokenizer {args.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_dataset, val_dataset = load_data(args, tokenizer)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=TRAINING_CONFIG['num_workers'], 
        pin_memory=TRAINING_CONFIG['pin_memory']
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    print(f"Configuring SOREModel {args.model_version}...")
    if args.model_version == 'v4':
        ConfigClass = ConfigV4
        ModelClass = SOREModel_v4
    else:
        ConfigClass = ConfigV3
        ModelClass = SOREModel_v3
        
    config = ConfigClass(
        vocab_size=len(tokenizer),
        context_size=args.context_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_alibi=(not args.no_alibi),
        use_rmsnorm=(not args.no_rmsnorm) # v4 default True
    )
    
    model = ModelClass(config)
    
    trainer = Trainer(model, tokenizer, args)
    
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
        
    trainer.train(train_loader, args.epochs, val_loader=val_loader)
    
    trainer.save_checkpoint(f'final_model_{args.stage}')

if __name__ == '__main__':
    main()

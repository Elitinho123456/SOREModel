
"""
distill_sore.py
Script to run Knowledge Distillation (Teacher -> Student).
Uses TeacherClient (OpenAI/Gemini/Local) and DistillationTrainer.
"""
import sys
import os
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.soreModel_v4_1 import SOREModel_v4_1, ModelConfig
from src.core.teacher_client import get_teacher_client
from src.training.distillation_trainer import DistillationTrainer
from src.data.dataset import TextDataset, InstructionDataset # Use appropriate dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Distill SOREModel v4')
    
    # Teacher Config
    parser.add_argument('--teacher_provider', type=str, required=True, choices=['openai', 'gemini'], help="Teacher API Provider")
    parser.add_argument('--teacher_model', type=str, default='gpt-3.5-turbo', help="Teacher Model Name")
    parser.add_argument('--api_key', type=str, default=None, help="API Key (or use env vars)")
    
    # Student Config
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset with prompts")
    parser.add_argument('--dataset_split', type=str, default='train')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_distill')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2')
    
    # Distillation Params
    parser.add_argument('--distill_alpha', type=float, default=0.5, help="Weight of LM loss vs KD loss (if logit based)")
    parser.add_argument('--distill_temperature', type=float, default=1.0)
    
    # Training Params
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--context_size', type=int, default=1024)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--use_amp', action='store_true', help="Use Mixed Precision")

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Init Teacher
    print(f"Initializing Teacher ({args.teacher_provider})...")
    teacher = get_teacher_client(
        args.teacher_provider, 
        model_name=args.teacher_model, 
        api_key=args.api_key
    )
    
    # Init Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Data
    # For distillation, we ideally have prompts. If dataset has 'text', we treat it as ground truth 
    # but we might want to augment. For this script, we assume standard training loop using DistillationTrainer
    # where the 'teacher' might be queried locally or text is already present.
    # NOTE: As per `distillation_trainer.py`, we primarily support offline/sequence level for API teachers.
    
    print(f"Loading dataset {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    
    # Simplifying: Assume text dataset for now, or instructions
    texts = [t for t in dataset['text'] if t.strip()] if 'text' in dataset.features else []
    # If no 'text', try custom handling...
    
    train_ds = TextDataset(texts, tokenizer, args.context_size)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # Init Student
    print("Initializing Student (SOREModel v4.1)...")
    config = ModelConfig(
        vocab_size=len(tokenizer),
        context_size=args.context_size,
        embed_dim=768, # Default small
        num_layers=8,  # Slightly smaller for student? or standard v4 size
        num_heads=8
    )
    student_model = SOREModel_v4_1(config)
    
    # Trainer
    trainer = DistillationTrainer(student_model, tokenizer, teacher, args)
    
    print("Starting Distillation Loop...")
    trainer.train(train_loader, args.epochs)
    
    trainer.save_checkpoint('distilled_model_final')

if __name__ == '__main__':
    main()

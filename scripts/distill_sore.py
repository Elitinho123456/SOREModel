
"""
distill_sore.py
Script to run Knowledge Distillation (Teacher -> Student).
Uses TeacherClient (OpenAI/Gemini/Local) and DistillationTrainer.
"""
import sys
import os
import json
import argparse
from pathlib import Path
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
    parser.add_argument('--teacher_provider', type=str, default='ollama',
                        choices=['openai', 'gemini', 'ollama'],
                        help="Teacher provider (default: 'ollama' — local, gratuito, sem rate limit)")
    parser.add_argument('--teacher_model', type=str, default='gemma3:12b',
                        help="Teacher model name (default: 'gemma3:12b' para Ollama)")
    parser.add_argument('--api_key', type=str, default=None,
                        help="API Key para openai/gemini (ou use env vars OPENAI_API_KEY / GEMINI_API_KEY)")
    parser.add_argument('--ollama_url', type=str, default='http://localhost:11434',
                        help="URL do servidor Ollama (default: http://localhost:11434)")
    
    # Student Config — provide either --dataset_path (local JSONL) or --dataset_name (HuggingFace hub)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--dataset_path', type=str, default=None, help="Path to a local .jsonl file")
    dataset_group.add_argument('--dataset_name', type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument('--dataset_split', type=str, default='train')
    parser.add_argument('--text_field', type=str, default='response', help="JSON field to use as text (default: 'response'). Falls back to 'text' if empty.")
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
    print(f"Initializing Teacher ({args.teacher_provider} / {args.teacher_model})...")
    teacher_kwargs = dict(model_name=args.teacher_model)
    if args.teacher_provider == 'ollama':
        teacher_kwargs['base_url'] = args.ollama_url
        # Faz health check antes de comecar — evita treinar e so descobrir o erro horas depois
        from src.core.teacher_client import OllamaTeacherClient
        _probe = OllamaTeacherClient(model_name=args.teacher_model, base_url=args.ollama_url)
        if not _probe.health_check():
            raise RuntimeError(
                f"Ollama nao esta acessivel em '{args.ollama_url}' ou o modelo '{args.teacher_model}' "
                "nao esta disponivel.\n"
                "Execute:\n"
                "  ollama serve          # inicia o servidor\n"
                f"  ollama pull {args.teacher_model}  # baixa o modelo"
            )
        print(f"  [OK] Ollama respondendo em {args.ollama_url} com '{args.teacher_model}'.")
    else:
        teacher_kwargs['api_key'] = args.api_key
    teacher = get_teacher_client(args.teacher_provider, **teacher_kwargs)
    
    # Init Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Data
    # Supports either a local JSONL file (--dataset_path) or a HuggingFace dataset (--dataset_name).
    # Use --text_field to specify which JSON key holds the text (default: 'response').
    # NOTE: As per `distillation_trainer.py`, we primarily support offline/sequence level for API teachers.

    def _extract_texts(records_or_dataset, field, is_hf=False):
        """Extract and filter non-empty strings from field, with 'text' as fallback."""
        fallback = 'text' if field != 'text' else 'response'
        if is_hf:
            features = records_or_dataset.features
            chosen = field if field in features else (fallback if fallback in features else None)
            if chosen is None:
                raise ValueError(f"Neither '{field}' nor '{fallback}' found in dataset features: {list(features.keys())}")
            if chosen != field:
                print(f"[WARN] Field '{field}' not found in dataset, falling back to '{chosen}'.")
            return [r for r in records_or_dataset[chosen] if r and str(r).strip()]
        else:
            texts = [str(r[field]) for r in records_or_dataset if field in r and str(r[field]).strip()]
            if not texts:
                print(f"[WARN] Field '{field}' not found or empty in JSONL, falling back to '{fallback}'.")
                texts = [str(r[fallback]) for r in records_or_dataset if fallback in r and str(r[fallback]).strip()]
            if not texts:
                raise ValueError(f"Neither '{field}' nor '{fallback}' found in JSONL records.")
            return texts

    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {dataset_path}")
        print(f"Loading local JSONL dataset from {dataset_path} (field: '{args.text_field}')...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]
        texts = _extract_texts(records, args.text_field, is_hf=False)
    else:
        from datasets import load_dataset
        print(f"Loading HuggingFace dataset '{args.dataset_name}' (field: '{args.text_field}')...")
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        texts = _extract_texts(dataset, args.text_field, is_hf=True)

    print(f"Loaded {len(texts)} samples.")
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

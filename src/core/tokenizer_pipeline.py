
"""
tokenizer_pipeline.py
Creates a BPE tokenizer with HuggingFace tokenizers library and optionally saves it.
Also provides quick wrappers for encoding/decoding and saving vocab.

Usage:
python tokenizer_pipeline.py --dataset_texts ./data/texts.txt --vocab_size 52000 --save_dir ./tokenizer
Or use HuggingFace datasets and build tokenizer from dataset texts.
"""
import argparse
from pathlib import Path

def build_and_save_tokenizer(text_paths, vocab_size=52000, save_dir="./tokenizer"):
    try:
        from tokenizers import Tokenizer, trainers, models, pre_tokenizers, processors
    except Exception as e:
        raise RuntimeError("Please install the 'tokenizers' package (huggingface/tokenizers).")

    files = []
    for p in text_paths:
        p = Path(p)
        if p.is_dir():
            files.extend([str(x) for x in p.glob("**/*.txt")])
        elif p.is_file():
            files.append(str(p))
    if len(files) == 0:
        raise RuntimeError("No text files found to train tokenizer.")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["", "<|pad|>"])
    tokenizer.train(files, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(Path(save_dir) / "tokenizer.json"))
    print("Saved tokenizer to", save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts", nargs="+", required=True, help="Paths to text files or directories")
    parser.add_argument("--vocab_size", type=int, default=52000)
    parser.add_argument("--save_dir", type=str, default="./tokenizer")
    args = parser.parse_args()
    build_and_save_tokenizer(args.texts, vocab_size=args.vocab_size, save_dir=args.save_dir)

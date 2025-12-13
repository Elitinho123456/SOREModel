import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    Dataset that tokenizes text on demand using a Hugging Face tokenizer.
    """
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        if not text or not text.strip():
            text = self.tokenizer.eos_token or "" 

        # Tokenize the text
        output = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Return input_ids, removing batch dimension
        return output['input_ids'].squeeze(0)

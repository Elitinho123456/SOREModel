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

class InstructionDataset(Dataset):
    """
    Dataset for Instruction Tuning (SFT).
    Formats input as: <user>\n{instruction}\n\n<assistant>\n{output}
    """
    def __init__(self, data, tokenizer, max_length):
        """
        data: list of dicts with keys 'instruction', 'input' (optional), 'output'
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get('instruction', '')
        inp = item.get('input', '')
        output = item.get('output', '')
        
        # Simple template
        if inp:
            prompt = f"<user>\n{instruction}\nInput: {inp}\n\n<assistant>\n{output}"
        else:
            prompt = f"<user>\n{instruction}\n\n<assistant>\n{output}"
            
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # For Causal LM, labels are usually same as input_ids
        # Advanced SFT masks the user part in labels, but for simplicity we train on all for now (or basic causal)
        input_ids = tokenized['input_ids'].squeeze(0)
        return input_ids

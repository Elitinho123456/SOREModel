import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

class Trainer:
    """
    Trainer class for SOREModel.
    Handles the training loop, evaluation, and checkpoint saving.
    """

    def __init__(self, model, tokenizer, args, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # State
        self.global_step = 0
        self.start_epoch = 0
        self.loss_history = []

    def get_scheduler(self, step):

        """Linear warmup and decay scheduler."""
        if step < self.args.warmup_steps:
            return self.args.learning_rate * (step / self.args.warmup_steps)
        
        total_decay_steps = 1000000 
        progress_decay = (step - self.args.warmup_steps) / max(1, total_decay_steps - self.args.warmup_steps)
        decay_factor = max(0.1, 1.0 - progress_decay)
        return self.args.learning_rate * decay_factor

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        current_lr = 0.0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch.to(self.device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.scaler is not None)):
                outputs = self.model(inputs)
                
                # Shift logits and labels for causal LM
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Normalize loss for gradient accumulation
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer Step
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                current_lr = self.get_scheduler(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging (wandb or others could go here)
            
            # Accumulate loss for display
            loss_item = loss.item() * self.args.gradient_accumulation_steps
            total_loss += loss_item
            avg_loss = total_loss / (batch_idx + 1)
            
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.2e}'})
            
            # Checkpoint capability inside epoch
            if self.global_step > 0 and self.global_step % self.args.save_steps == 0 and \
               (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}')
                
        return avg_loss

    def train(self, dataloader, epochs):
        print(f"Starting training on {self.device}...")
        
        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()
            
            avg_loss = self.train_epoch(dataloader, epoch)
            self.loss_history.append(avg_loss)
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch + 1}/{epochs} Completed - Avg Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s')
            
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}')

    def save_checkpoint(self, name):
        checkpoint_dir = Path(self.args.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = checkpoint_dir / 'model.pt'
        
        # Save model config explicitly if available
        model_config = getattr(self.model, 'cfg', None)
        model_config_dict = model_config.__dict__ if model_config else {}

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.global_step,
            'config': model_config_dict
        }, model_path)
        
        # Save training args
        config_path = checkpoint_dir / 'train_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:

            # Filter non-serializable args if any, but vars() usually ok for argparse
            json.dump(vars(self.args), f, indent=2, ensure_ascii=False)
            
        print(f"Checkpoint saved at {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path):
        
        """Resumes training from a checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        self.global_step = checkpoint.get('step', 0)
        return self.global_step
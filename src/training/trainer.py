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
        use_amp = getattr(args, 'use_amp', False) and (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler() if use_amp else None
        
        # State
        self.global_step = 0
        self.start_epoch = 0
        self.loss_history = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def get_scheduler(self, step, total_steps):
        """
        Support for different schedulers.
        """
        scheduler_type = getattr(self.args, 'lr_scheduler', 'cosine')
        warmup_steps = getattr(self.args, 'warmup_steps', 0)

        # Warmup phase
        if step < warmup_steps:
            return self.args.learning_rate * (step / max(1, warmup_steps))

        # Main schedule
        if scheduler_type == "cosine":
            # Cosine decay
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return self.args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        elif scheduler_type == "step":
            # Simple step decay (e.g. drop every 1 epoch equivalent?)
            # Simplified for now just keeping constant or check args
            return self.args.learning_rate
        else:
            return self.args.learning_rate

    def train_epoch(self, dataloader, epoch, total_steps):
        self.model.train()
        total_loss = 0.0
        current_lr = 0.0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch.to(self.device)
            
            # Forward pass with mixed precision
            use_amp = (self.scaler is not None)
            with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                # Trainer can handle both standard tuple outputs or just logits depending on model
                if hasattr(self, 'compute_loss'):
                     loss = self.compute_loss(self.model, inputs)
                else:
                    # Default LM behavior
                    outputs = self.model(inputs)
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = inputs[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                loss = loss / self.args.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer Step
            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                current_lr = self.get_scheduler(self.global_step, total_steps)
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
            
            # Accumulate loss
            loss_item = loss.item() * self.args.gradient_accumulation_steps
            total_loss += loss_item
            avg_loss = total_loss / (batch_idx + 1)
            
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.2e}'})
            
            if self.global_step > 0 and self.global_step % self.args.save_steps == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}')
                
        return avg_loss

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch.to(self.device)
                outputs = self.model(inputs)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, train_loader, epochs, val_loader=None):
        print(f"Starting training on {self.device}...")
        total_steps = len(train_loader) * epochs // self.args.gradient_accumulation_steps
        
        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()
            
            avg_loss = self.train_epoch(train_loader, epoch, total_steps)
            self.loss_history.append(avg_loss)
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s')
            
            # Validation & Early Stopping
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")
                
                if val_loss < (self.best_val_loss - getattr(self.args, 'min_delta', 0.0)):
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model')
                    self.patience_counter = 0
                    print("New best model saved!")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= getattr(self.args, 'early_stopping_patience', 3):
                        print("Early stopping triggered!")
                        break

            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}')

    def save_checkpoint(self, name):
        checkpoint_dir = Path(self.args.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / 'model.pt'
        
        model_config = getattr(self.model, 'cfg', None)
        model_config_dict = model_config.__dict__ if model_config else {}

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.global_step,
            'config': model_config_dict
        }, model_path)
        
        # Save training args as JSON
        config_path = checkpoint_dir / 'train_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            args_dict = vars(self.args)
            # handle non-serializable objects if any
            json.dump({k: str(v) for k,v in args_dict.items()}, f, indent=2)
            
        print(f"Checkpoint saved at {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('step', 0)
        return self.global_step
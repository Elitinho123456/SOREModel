import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm


class Trainer:
    """
    Trainer class for SOREModel.
    Handles the training loop, evaluation, and checkpoint saving.

    Checkpoint slots (max 3, each overwrites its previous version):
      - epoch_start    : saved at the *start* of every epoch
      - best_model     : saved whenever validation loss improves
      - auto_checkpoint: periodic save every `save_steps` global steps
    """

    # ------------------------------------------------------------------ #
    #  Init                                                                #
    # ------------------------------------------------------------------ #

    def __init__(self, model, tokenizer, args, device=None):
        self.model     = model
        self.tokenizer = tokenizer
        self.args      = args
        self.device    = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Mixed Precision Scaler
        use_amp = getattr(args, 'use_amp', False) and (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler() if use_amp else None

        # Training state
        self.global_step    = 0
        self.start_epoch    = 0
        self.loss_history   = []
        self.best_val_loss  = float('inf')
        self.patience_counter = 0

    # ------------------------------------------------------------------ #
    #  Checkpoint helpers                                                  #
    # ------------------------------------------------------------------ #

    def _checkpoint_dir(self, slot: str) -> Path:
        """Return (and create) the fixed directory for a checkpoint slot."""
        path = Path(self.args.output_dir) / slot
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_checkpoint(self, slot: str, extra_meta: dict = None):
        """
        Save model + optimizer state into one of the 3 fixed slots.
        Overwrites whatever was previously stored there.

        Args:
            slot       : 'epoch_start' | 'best_model' | 'auto_checkpoint'
                          (or any string for ad-hoc saves like 'final_model_*')
            extra_meta : optional dict merged into the saved payload
        """
        checkpoint_dir = self._checkpoint_dir(slot)
        model_path     = checkpoint_dir / 'model.pt'

        model_config      = getattr(self.model, 'cfg', None)
        model_config_dict = model_config.__dict__ if model_config else {}

        payload = {
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step':                 self.global_step,
            'epoch':                self.start_epoch,
            'best_val_loss':        self.best_val_loss,
            'config':               model_config_dict,
        }
        if extra_meta:
            payload.update(extra_meta)

        torch.save(payload, model_path)

        # Lightweight JSON with training args (always overwritten)
        config_path = checkpoint_dir / 'train_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(
                {k: str(v) for k, v in vars(self.args).items()},
                f, indent=2,
            )

        print(f"[Checkpoint] '{slot}' saved -> {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint with minimal VRAM impact.

        Root cause of VRAM overflow on resume
        ──────────────────────────────────────
        When `map_location=self.device` is used, *both* the new weights
        and the old weights that are already on the GPU coexist briefly,
        roughly doubling VRAM usage.  The optimizer state (moment tensors
        ~2× model size for Adam) also lands entirely on VRAM at the same
        time, causing a massive transient spike that can kill the process.

        Fix strategy
        ────────────
        1. Load the raw file entirely to **CPU** – zero VRAM consumed.
        2. Call `load_state_dict()` on the model (already on GPU) which
           does an **in-place** copy from CPU tensors, never creating a
           full GPU-side duplicate.
        3. Load the optimizer state keeping every tensor on CPU.
           PyTorch's AdamW will migrate individual tensors to the device
           lazily on the first `optimizer.step()`, so at most one
           parameter's state occupies extra VRAM at a time.
        4. Delete the raw checkpoint dict and call `empty_cache()` to
           reclaim any pinned host memory immediately.
        """
        print(f"[Checkpoint] Loading from '{checkpoint_path}' …")

        # 1 – Load everything to CPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 2 – Restore model weights in-place (avoids doubling VRAM)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 3 – Restore optimizer state and move tensors to the training device.
        #     NOTE: keeping them on CPU was intended to save VRAM, but PyTorch's
        #     multi-tensor Adam kernel (used with AMP) requires params AND their
        #     moments to be on the **same** device, causing:
        #       RuntimeError: Expected all tensors to be on the same device,
        #                     but found at least two devices, cuda:0 and cpu!
        #     We still load the full checkpoint to CPU first (step 1 above) so
        #     we never hold two copies of the model on VRAM simultaneously.
        #     The optimizer state is then transferred to the device one tensor at
        #     a time, which is far cheaper than doubling model VRAM.
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        # 4 – Restore training metadata
        self.global_step   = checkpoint.get('step', 0)
        self.start_epoch   = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # 5 – Free raw checkpoint and flush CUDA allocator
        del checkpoint
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        print(
            f"[Checkpoint] Resumed — step={self.global_step}, "
            f"epoch={self.start_epoch}, best_val_loss={self.best_val_loss:.4f}"
        )
        return self.global_step

    # ------------------------------------------------------------------ #
    #  Scheduler                                                           #
    # ------------------------------------------------------------------ #

    def get_scheduler(self, step, total_steps):
        """Compute the learning rate for the current step."""
        scheduler_type = getattr(self.args, 'lr_scheduler', 'cosine')
        warmup_steps   = getattr(self.args, 'warmup_steps', 0)

        # Warmup phase
        if step < warmup_steps:
            return self.args.learning_rate * (step / max(1, warmup_steps))

        # Main schedule
        if scheduler_type == 'cosine':
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return self.args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        elif scheduler_type == 'step':
            return self.args.learning_rate
        else:
            return self.args.learning_rate

    # ------------------------------------------------------------------ #
    #  Training loop                                                       #
    # ------------------------------------------------------------------ #

    def train_epoch(self, dataloader, epoch, total_steps):
        self.model.train()
        total_loss = 0.0
        current_lr = 0.0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch.to(self.device)

            # ── Forward pass ──────────────────────────────────────────
            use_amp = (self.scaler is not None)
            with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                if hasattr(self, 'compute_loss'):
                    loss = self.compute_loss(self.model, inputs)
                else:
                    outputs      = self.model(inputs)
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = inputs[..., 1:].contiguous()
                    loss_fct     = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                loss = loss / self.args.gradient_accumulation_steps

            # ── Backward pass ─────────────────────────────────────────
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # ── Optimizer step ────────────────────────────────────────
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

            # ── Metrics ───────────────────────────────────────────────
            loss_item   = loss.item() * self.args.gradient_accumulation_steps
            total_loss += loss_item
            avg_loss    = total_loss / (batch_idx + 1)

            progress_bar.set_postfix(
                {'loss': f'{avg_loss:.4f}', 'lr': f'{current_lr:.2e}'}
            )

            # ── Slot 3: auto_checkpoint (periodic) ────────────────────
            if self.global_step > 0 and self.global_step % self.args.save_steps == 0:
                self.save_checkpoint('auto_checkpoint')

        return avg_loss

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                inputs       = batch.to(self.device)
                outputs      = self.model(inputs)
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = inputs[..., 1:].contiguous()
                loss_fct     = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, train_loader, epochs, val_loader=None):
        print(f"Starting training on {self.device}...")
        total_steps = (
            len(train_loader) * epochs // self.args.gradient_accumulation_steps
        )

        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()

            # ── Slot 1: epoch_start ───────────────────────────────────
            self.start_epoch = epoch
            self.save_checkpoint('epoch_start')

            avg_loss = self.train_epoch(train_loader, epoch, total_steps)
            self.loss_history.append(avg_loss)

            epoch_time = time.time() - start_time
            print(
                f'Epoch {epoch + 1}/{epochs} — '
                f'Loss: {avg_loss:.4f} — Time: {epoch_time:.2f}s'
            )

            # ── Validation & Early Stopping ───────────────────────────
            if val_loader:
                val_loss = self.validate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")

                if val_loss < (self.best_val_loss - getattr(self.args, 'min_delta', 0.0)):
                    self.best_val_loss = val_loss
                    # ── Slot 2: best_model ────────────────────────────
                    self.save_checkpoint('best_model')
                    self.patience_counter = 0
                    print("New best model saved!")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= getattr(self.args, 'early_stopping_patience', 3):
                        print("Early stopping triggered!")
                        break
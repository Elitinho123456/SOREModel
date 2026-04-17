"""
dataset.py
Datasets do SOREModel.

TextDataset        — Pre-training / Sequence-Level Distillation
InstructionDataset — Instruction Fine-Tuning (SFT) com mascara de labels no prompt

BUG CORRIGIDO (InstructionDataset):
  O dataset anterior retornava apenas input_ids e treinava o modelo sobre TODO o texto,
  inclusive o trecho do usuario (instrucao). Isso e ineficiente: o modelo nao precisa
  aprender a repetir a instrucao, so a resposta.

  A correcao retorna um dict {'input_ids', 'labels'} onde as posicoes correspondentes
  ao prompt do usuario sao marcadas com -100 (ignoradas pelo CrossEntropyLoss).

NOTA PARA O TRAINER:
  Quando InstructionDataset for usado, train_epoch() recebe um dict por batch.
  O DistillationTrainer aceita dict nativamente. Para o Trainer base, use o
  collate_fn padrao do PyTorch com default_collate, que agrupara os dicts corretamente.
"""
import torch
from torch.utils.data import Dataset


_IGNORE_INDEX = -100  # Padrao do CrossEntropyLoss para ignorar tokens


class TextDataset(Dataset):
    """
    Dataset para pre-treino e Sequence-Level Distillation.
    Retorna apenas input_ids (sem mascara de labels).
    """

    def __init__(self, texts: list, tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.texts[idx]
        if not text or not text.strip():
            text = getattr(self.tokenizer, "eos_token", "") or ""

        output = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return output["input_ids"].squeeze(0)  # (T,)


class InstructionDataset(Dataset):
    """
    Dataset para Instruction Fine-Tuning (SFT) com mascara de labels.

    Formato do prompt:
        <user>
        {instruction}
        Input: {input}    (opcional)

        <assistant>
        {output}

    Retorna um dict:
        {
          'input_ids': LongTensor (T,),
          'labels':    LongTensor (T,)   -- regioes do prompt = -100
        }

    Isso garante que o modelo so aprende a GERAR a resposta do assistente,
    nao a repetir a instrucao/prompt.
    """

    def __init__(self, data: list, tokenizer, max_length: int):
        """
        Args:
            data:       Lista de dicts com chaves 'instruction', 'input' (opcional), 'output'.
            tokenizer:  Tokenizador HuggingFace.
            max_length: Comprimento maximo da sequencia completa (prompt + resposta).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")

        # Monta a parte do prompt (usuario) separadamente para calcular o comprimento
        if inp:
            prompt_text = f"<user>\n{instruction}\nInput: {inp}\n\n<assistant>\n"
        else:
            prompt_text = f"<user>\n{instruction}\n\n<assistant>\n"

        full_text = prompt_text + output

        # Tokeniza o texto completo
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"].squeeze(0)  # (T,)

        # Tokeniza apenas o prompt para descobrir onde a resposta comeca
        # (sem padding para ter o tamanho real)
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        # Labels: -100 nas posicoes do prompt, ids reais nas posicoes da resposta
        labels = input_ids.clone()
        labels[:prompt_len] = _IGNORE_INDEX

        # Tokens de padding tambem devem ser ignorados
        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[input_ids == pad_id] = _IGNORE_INDEX

        return {
            "input_ids": input_ids,   # (T,)
            "labels": labels,         # (T,) com -100 no prompt e no padding
        }


class DistillationDataset(Dataset):
    """
    Dataset para Sequence-Level Distillation com proporcao controlada de dados.

    Mistura dados gerados pelo teacher com dados humanos frescos na proporcao
    definida pelo AZR Orchestrator (padrao: 70% teacher / 30% humano).
    Previne Model Collapse conforme especificado no plano AZR.

    Args:
        teacher_texts:  Lista de strings geradas pelo teacher (via TeacherClient.generate).
        human_texts:    Lista de strings de dados humanos frescos (papers, forums, etc).
        tokenizer:      Tokenizador.
        max_length:     Comprimento maximo de sequencia.
        teacher_ratio:  Proporcao de dados do teacher no dataset final (0.0 a 1.0).
    """

    def __init__(
        self,
        teacher_texts: list,
        human_texts: list,
        tokenizer,
        max_length: int,
        teacher_ratio: float = 0.70,
    ):
        if not 0.0 <= teacher_ratio <= 1.0:
            raise ValueError(f"teacher_ratio deve estar em [0, 1]. Recebido: {teacher_ratio}")

        n_total = len(teacher_texts) + len(human_texts)
        n_teacher_target = int(n_total * teacher_ratio)

        # Limita ou repete dados para atingir a proporcao desejada
        teacher_sample = teacher_texts[:n_teacher_target]
        n_human_target = n_total - len(teacher_sample)
        human_sample = human_texts[:n_human_target]

        all_texts = teacher_sample + human_sample

        self._inner = TextDataset(all_texts, tokenizer, max_length)

        # Metadados para auditoria do Orchestrator
        self.n_teacher = len(teacher_sample)
        self.n_human = len(human_sample)
        self.actual_teacher_ratio = self.n_teacher / max(len(all_texts), 1)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._inner[idx]

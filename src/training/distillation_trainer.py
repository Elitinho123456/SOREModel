"""
distillation_trainer.py
Trainer de Destilação de Conhecimento (Knowledge Distillation) para o SOREModel.

Suporta três modos de distilação:

  'sequence'  ─ Sequence-Level KD (padrão)
    O teacher já foi usado OFFLINE para gerar as respostas que estão no dataset.
    O student treina via cross-entropy contra esses dados.
    Use quando o teacher é uma API (Gemini, GPT) que não expõe logits.
    Custo zero de latência na hora do treino.

  'logit'     ─ Token-Level KD via KLDiv (Hinton et al., 2015)
    O teacher é chamado ONLINE por batch para retornar logits brutos.
    Loss = alpha * CE(student, labels) + (1-alpha) * T² * KLDiv(student ‖ teacher)
    Use com teacher LOCAL (vLLM, HuggingFace local, Ollama).
    Mais rico em "dark knowledge" — o student aprende a distribuição inteira, não
    só o token correto.

  'hybrid'    ─ Tenta modo 'logit'; se get_logits() retornar None, cai para 'sequence'.

Boas práticas de ancoragem semântica (prevenção de Model Collapse AZR):
  - Dataset: 70% dados do teacher + 30% dados humanos frescos.
  - Os Gabaritos de Ouro do Watchdog NUNCA entram no dataset de treino.
  - A validação usa apenas LM Loss (sem teacher) para comparação limpa entre checkpoints.

PROBLEMAS CORRIGIDOS NESTA VERSÃO:
  1. compute_loss chamava model(inputs) sem desempacotar — o SOREModel_v4_1 retorna
     (logits, kv_cache) quando use_cache=True, e logits simples quando use_cache=False.
     O trainer chama com use_cache=False, então o retorno já é o tensor correto, mas
     adicionamos uma guarda explícita para nunca travar silenciosamente.
  2. A divisão da loss pelo gradient_accumulation_steps é feita no Trainer pai (train_epoch).
     compute_loss NÃO divide — a divisão dupla causaria gradientes 2x menores.
  3. _fetch_teacher_logits fazia batch_decode sem checar se self.tokenizer existe.
  4. Pad de logits do teacher usava torch.zeros — correto apenas se os tokens padded
     forem mascarados. Adicionamos máscara logit (fill -inf) para tokens de padding.
  5. InstructionDataset treina sobre TUDO (inclusive prompt do usuário). Adicionado
     suporte a retornar labels com máscara (-100) no trecho do prompt para SFT correto.
  6. O validate() do DistillationTrainer sobrescrevia o do Trainer mas duplicava código.
     Refatorado para reutilizar _lm_loss().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Literal, Optional

from .trainer import Trainer

log = logging.getLogger("AZR.DistillationTrainer")

DistillationMode = Literal["sequence", "logit", "hybrid"]

# Índice ignorado pela CrossEntropy (tokens de padding / região de prompt no SFT)
_IGNORE_INDEX = -100


class DistillationTrainer(Trainer):
    """
    Estende o Trainer base para suporte a Knowledge Distillation.

    Uso básico (modo sequence — teacher via API):
        trainer = DistillationTrainer(
            model=student,
            tokenizer=tokenizer,
            teacher_client=gemini_client,   # ou None se dados já estão no dataset
            args=args,
            distillation_mode="sequence",
        )
        trainer.train(train_loader, epochs=3, val_loader=val_loader)

    Uso avançado (modo logit — teacher local):
        trainer = DistillationTrainer(
            model=student,
            tokenizer=tokenizer,
            teacher_client=local_vllm_client,
            args=args,
            distillation_mode="logit",
        )
    """

    def __init__(
        self,
        model,
        tokenizer,
        teacher_client,
        args,
        device=None,
        distillation_mode: DistillationMode = "sequence",
    ):
        super().__init__(model, tokenizer, args, device)
        self.teacher = teacher_client
        self.distillation_mode: DistillationMode = distillation_mode

        # alpha: peso da LM Loss.  (1-alpha): peso da KD Loss.
        # alpha=1.0 → distilação pura de sequência (sem KLDiv).
        # alpha=0.5 → equilíbrio (recomendado para 'logit').
        self.alpha: float = float(getattr(args, "distill_alpha", 0.5))
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"distill_alpha deve estar em [0, 1]. Recebido: {self.alpha}")

        # Temperatura T > 1 suaviza a distribuição do teacher, transferindo mais
        # "dark knowledge". Hinton et al. recomendam T=3-5 para tarefas de NLP.
        self.temperature: float = max(1.0, float(getattr(args, "distill_temperature", 2.0)))

        log.info(
            f"DistillationTrainer | mode={distillation_mode} "
            f"| alpha={self.alpha} | T={self.temperature} | device={self.device}"
        )

    # ------------------------------------------------------------------ #
    #  Extração segura de logits do modelo                                 #
    # ------------------------------------------------------------------ #

    def _get_student_logits(self, model, inputs: torch.Tensor) -> torch.Tensor:
        """
        Roda forward pass do student com use_cache=False e retorna apenas os logits.

        O SOREModel_v4_1.forward() retorna:
          - logits: Tensor (B, T, V)              quando use_cache=False
          - (logits, kv_list): tuple               quando use_cache=True

        O trainer SEMPRE usa use_cache=False durante treino — só geração usa o cache.
        A guarda abaixo previne bugs silenciosos caso alguém mude o modelo.
        """
        out = model(inputs)  # use_cache=False por padrão
        if isinstance(out, tuple):
            logits = out[0]   # desempacota caso o modelo retorne (logits, kv)
        else:
            logits = out
        return logits  # (B, T, V)

    # ------------------------------------------------------------------ #
    #  Loss functions                                                      #
    # ------------------------------------------------------------------ #

    def _lm_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Cross-Entropy de Language Modeling (Causal LM, shift de 1).

        Args:
            logits: (B, T, V) — logits do student
            labels: (B, T)   — token ids  (usa _IGNORE_INDEX=-100 para mascarar padding/prompt)

        O shift faz o modelo predizer o token t+1 a partir do token t.
        Tokens marcados com -100 em `labels` são ignorados pelo CrossEntropy.
        """
        shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()        # (B, T-1)

        loss_fct = nn.CrossEntropyLoss(ignore_index=_IGNORE_INDEX)
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def _kl_div_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        KL Divergence Loss: KLDiv(P_teacher ‖ Q_student) com temperatura T.

        Multiplicado por T² para preservar a escala do gradiente com temperatura
        alta (Hinton et al., 2015, eq. 4).

        IMPORTANTE: a KLDiv é calculada nas posições shifted (mesmo que a LM Loss)
        para manter alinhamento. Posições de padding (-100) são mascaradas.

        Args:
            student_logits: (B, T, V)
            teacher_logits: (B, T, V)  — deve ter o mesmo vocabulário do student
            labels:         (B, T)     — opcional; usado para mascarar padding

        Returns:
            Scalar tensor com a KD loss.
        """
        T = self.temperature

        # Shift para alinhar com a LM Loss
        s_logits = student_logits[:, :-1, :].contiguous()  # (B, T-1, V)
        t_logits = teacher_logits[:, :-1, :].contiguous()  # (B, T-1, V)

        s_log_probs = F.log_softmax(s_logits / T, dim=-1)  # (B, T-1, V)
        t_probs     = F.softmax(t_logits    / T, dim=-1)   # (B, T-1, V)

        # Máscara de padding: não queremos aprender a imitar o teacher em tokens -100
        if labels is not None:
            shift_labels = labels[:, 1:]                    # (B, T-1)
            non_pad_mask = (shift_labels != _IGNORE_INDEX)  # (B, T-1) bool
            s_log_probs = s_log_probs[non_pad_mask]         # (N_valido, V)
            t_probs     = t_probs[non_pad_mask]             # (N_valido, V)

        kl = F.kl_div(
            s_log_probs.view(-1, s_log_probs.size(-1)),
            t_probs.view(-1, t_probs.size(-1)),
            reduction="batchmean",
        )
        return kl * (T ** 2)

    # ------------------------------------------------------------------ #
    #  compute_loss — ponto de entrada do training loop do Trainer pai     #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        model,
        inputs: torch.Tensor,
        return_outputs: bool = False,
    ):
        """
        Calcula a loss combinada conforme o modo de distilação.

        NOTA: NÃO divida por gradient_accumulation_steps aqui.
        Isso é feito no train_epoch() do Trainer pai, logo após a chamada.

        Args:
            model:          O student model.
            inputs:         Tensor (B, T) de token ids. Labels = inputs (Causal LM).
                            Para SFT com máscara, o dataset deve retornar um dict com
                            {'input_ids': ..., 'labels': ...} e este método ser adaptado.
            return_outputs: Se True, retorna (loss, logits).

        Returns:
            loss scalar, ou (loss scalar, logits) se return_outputs=True.
        """
        student_logits = self._get_student_logits(model, inputs)  # (B, T, V)

        # ── Modo Sequência ──────────────────────────────────────────────────
        if self.distillation_mode == "sequence":
            loss = self._lm_loss(student_logits, inputs)
            return (loss, student_logits) if return_outputs else loss

        # ── Modo Logit / Hybrid ─────────────────────────────────────────────
        lm_loss = self._lm_loss(student_logits, inputs)
        teacher_logits = self._fetch_teacher_logits(inputs)

        if teacher_logits is None:
            if self.distillation_mode == "hybrid":
                log.debug("teacher_logits indisponível (modo hybrid) — usando apenas LM Loss.")
                loss = lm_loss
            else:
                # Falha explícita: modo 'logit' exige teacher com logits reais
                raise RuntimeError(
                    "Modo 'logit' exige um TeacherClient com get_logits() funcional. "
                    "O cliente atual retornou None. Use modo 'sequence' para APIs externas."
                )
        else:
            kd_loss = self._kl_div_loss(student_logits, teacher_logits, labels=inputs)
            loss = self.alpha * lm_loss + (1.0 - self.alpha) * kd_loss
            log.debug(
                f"LM={lm_loss.item():.4f} | KD={kd_loss.item():.4f} | "
                f"Total={loss.item():.4f} | alpha={self.alpha}"
            )

        return (loss, student_logits) if return_outputs else loss

    # ------------------------------------------------------------------ #
    #  Teacher logits (para modo logit / hybrid)                          #
    # ------------------------------------------------------------------ #

    def _fetch_teacher_logits(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Obtém logits do teacher para os inputs do batch.

        Para teachers via API (Gemini, OpenAI): get_logits() retorna None.
        Para teachers locais (vLLM, HF local): retorna tensor de logits.

        O padding é feito com -inf (não com zeros) para que softmax sobre posições
        padded resulte em distribuição uniforme → contribuição mínima na KLDiv.

        Returns:
            Tensor (B, T, V) no mesmo device do student, ou None.
        """
        if self.teacher is None or self.tokenizer is None:
            return None

        try:
            texts = self.tokenizer.batch_decode(inputs, skip_special_tokens=True)

            all_logits: list[torch.Tensor] = []
            for text in texts:
                raw = self.teacher.get_logits(text)
                if raw is None:
                    return None   # API não suporta logits — sinaliza degradação
                if not isinstance(raw, torch.Tensor):
                    raw = torch.tensor(raw, dtype=torch.float32)
                if raw.dim() == 1:
                    raw = raw.unsqueeze(0)  # (1, V) → compatibilidade com alguns teachers
                all_logits.append(raw)  # (T_i, V)

            # Pad com -inf para que softmax de tokens padded → uniforme (~0 KL contribuição)
            max_t = max(t.shape[0] for t in all_logits)
            V = all_logits[0].shape[-1]
            padded = torch.full((len(all_logits), max_t, V), fill_value=float("-inf"))
            for i, t_logits in enumerate(all_logits):
                t = t_logits.shape[0]
                padded[i, :t, :] = t_logits

            return padded.to(self.device)

        except Exception as e:
            log.warning(f"_fetch_teacher_logits falhou: {e}. Degradando para LM Loss.")
            return None

    # ------------------------------------------------------------------ #
    #  Validação                                                           #
    # ------------------------------------------------------------------ #

    def validate(self, dataloader) -> float:
        """
        Validação usando apenas LM Loss (sem teacher).

        Por que não usar KLDiv na validação?
          - Chamar o teacher por batch na validação pode ser lento/caro.
          - A LM Loss é uma métrica universal e comparável entre todos os checkpoints.
          - A perplexidade derivada da LM Loss é o benchmark padrão para LLMs.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch.to(self.device)
                logits = self._get_student_logits(self.model, inputs)
                loss = self._lm_loss(logits, inputs)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

"""
Gabaritos de Ouro do Watchdog — NAO incluir no dataset de treino.

Qualquer falha aqui = Morte Cognitiva = Rollback automatico.
"""

# ─── Teste 1: Matematica basica ───────────────────────────────────────────────
assert 2 + 2 == 4, "Matematica basica falhou"

# ─── Teste 2: Modulos AZR importaveis ─────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from src.azr.core.telemetry import Telemetry
from src.azr.core.orchestrator import Orchestrator
from src.azr.security.snapshot import SnapshotManager
from src.azr.security.watchdog import Watchdog
from src.azr.security.auditor import SecurityAuditor

assert Telemetry is not None, "Telemetry nao importou"
assert Orchestrator is not None, "Orchestrator nao importou"

# ─── Teste 3: Auditor detecta while True ──────────────────────────────────────
auditor = SecurityAuditor()
result = auditor.audit("while True:\n    pass")
assert not result.passed, "Auditor deveria barrar while True sem break"
assert any(v.rule == "INFINITE_LOOP" for v in result.violations), "Regra INFINITE_LOOP nao disparou"

# ─── Teste 4: Auditor aprova codigo legitimo ──────────────────────────────────
ok_result = auditor.audit("x = 1 + 1\nassert x == 2")
assert ok_result.passed, f"Auditor rejeitou codigo legitimo: {ok_result.violations}"

# ─── Teste 5: Orchestrator cooldown ───────────────────────────────────────────
orch = Orchestrator(failure_log_path="/tmp/test_azr_failure.json")
orch.add_cooldown("test_component", duration_sec=9999)
allowed, reason = orch.can_run_experiment("test_component")
assert not allowed, "Cooldown deveria bloquear execucao"
assert "cooldown" in reason.lower(), f"Mensagem de cooldown inesperada: {reason}"

# ─── Teste 6: Pipeline de Distilacao (sem GPU) ────────────────────────────────
import torch
from src.models.soreModel_v4_1 import SOREModel_v4_1, ModelConfig
from src.training.distillation_trainer import DistillationTrainer
from src.data.dataset import DistillationDataset, InstructionDataset

# -- Modelo minusculo so para garantir que o forward/loss funcionam
cfg = ModelConfig(num_layers=2, embed_dim=64, num_heads=4,
                  vocab_size=100, context_size=16, dropout=0.0)
student = SOREModel_v4_1(cfg)

class _FakeArgs:
    distill_alpha        = 0.5
    distill_temperature  = 2.0
    learning_rate        = 1e-3
    weight_decay         = 0.01
    use_amp              = False
    gradient_accumulation_steps = 1
    save_steps           = 9999
    output_dir           = "/tmp/azr_golden_ckpt"

trainer = DistillationTrainer(
    student, None, None, _FakeArgs(), distillation_mode="sequence"
)
batch = torch.randint(0, 100, (2, 16))
loss = trainer.compute_loss(student, batch)
assert isinstance(loss, torch.Tensor), "compute_loss deve retornar um Tensor"
assert loss.item() > 0, f"Loss deve ser positivo, recebeu {loss.item()}"
assert not torch.isnan(loss), "Loss nao pode ser NaN"
assert not torch.isinf(loss), "Loss nao pode ser Inf"

# -- DistillationDataset (verificar proporcao teacher/humano)
ds = DistillationDataset(
    teacher_texts=["texto do teacher"] * 70,
    human_texts=["texto humano"] * 30,
    tokenizer=None,   # nao usado neste assert
    max_length=16,
    teacher_ratio=0.70,
)
assert ds.n_teacher == 70, f"Esperado 70 samples do teacher, recebeu {ds.n_teacher}"
assert abs(ds.actual_teacher_ratio - 0.70) < 0.05, "Proporcao teacher/humano fora do esperado"

print("Todos os gabaritos de ouro passaram.")

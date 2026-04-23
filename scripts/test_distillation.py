"""
test_distillation.py
====================
Smoke-test da pipeline de Knowledge Distillation do SOREModel.

Usa o Ollama local (gemma4) como teacher real e um student minimo para
verificar que toda a pipeline funciona de ponta a ponta rapidamente.

Executar:
    python scripts/test_distillation.py
    python scripts/test_distillation.py --mode sequence
    python scripts/test_distillation.py --mode hybrid
    python scripts/test_distillation.py --ollama_model gemma4:latest
"""
import sys
import argparse
import traceback
from pathlib import Path

import torch

# -- Adiciona raiz do projeto ao path -----------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer
from src.models.soreModel_v4_1 import SOREModel_v4_1, ModelConfig
from src.core.teacher_client import OllamaTeacherClient, get_teacher_client
from src.training.distillation_trainer import DistillationTrainer
from src.data.dataset import TextDataset
from torch.utils.data import DataLoader


# -- Textos sinteticos para o teste -------------------------------------------
SAMPLE_TEXTS = [
    "Knowledge distillation transfers information from a large teacher model to a smaller student model.",
    "Transformers have revolutionized natural language processing tasks significantly.",
    "The SOREModel is a lightweight language model trained with knowledge distillation.",
    "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
    "Gradient accumulation helps train larger models on limited GPU memory efficiently.",
    "Mixed precision training with AMP reduces memory usage and speeds up training.",
    "The learning rate scheduler controls how the learning rate changes during training.",
    "Deep learning models require large amounts of data to generalize well to new examples.",
]


# -- Args falsos (substitui o argparse do distill_sore.py) --------------------
class FakeArgs:
    output_dir                  = "./checkpoints_distill_test"
    learning_rate               = 3e-4
    weight_decay                = 0.01
    use_amp                     = False
    save_steps                  = 9999   # nao salva durante o smoke-test
    warmup_steps                = 1
    gradient_accumulation_steps = 1
    distill_alpha               = 0.5
    distill_temperature         = 2.0
    lr_scheduler                = "cosine"
    early_stopping_patience     = 3
    min_delta                   = 0.0


# -- Helpers ------------------------------------------------------------------
def sep(title=""):
    line = "-" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def check(cond: bool, msg: str):
    status = "[OK]   " if cond else "[FAIL] "
    print(f"  {status}{msg}")
    if not cond:
        raise AssertionError(f"Verificacao falhou: {msg}")


# -- Teste principal ----------------------------------------------------------
def run_test(mode: str, ollama_url: str, ollama_model: str) -> bool:
    sep(f"Modo: '{mode}' | Teacher: {ollama_model} via Ollama")

    # 1. Health-check do Ollama
    print("  [1] Verificando conexao com Ollama...")
    teacher = OllamaTeacherClient(model_name=ollama_model, base_url=ollama_url)
    ok = teacher.health_check()
    check(ok, f"Ollama acessivel em {ollama_url} com modelo '{ollama_model}'")

    # 2. Geracao de texto pelo teacher (smoke-test da API)
    print("  [2] Testando teacher.generate()...")
    resposta = teacher.generate(
        "Em uma frase, o que e Knowledge Distillation?",
        max_tokens=80,
        temperature=0.3,
    )
    check(isinstance(resposta, str) and len(resposta) > 0,
          f"Teacher gerou texto ({len(resposta)} chars)")
    print(f"       Teacher: \"{resposta[:100].strip()}\"")

    # 3. get_logits() -- deve retornar None (Ollama nao expoe logits brutos)
    print("  [3] Verificando get_logits() do Ollama...")
    logits = teacher.get_logits("teste")
    check(logits is None, "get_logits() retorna None (esperado para Ollama)")

    # 4. Tokenizer
    print("  [4] Carregando tokenizer gpt2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    check(tokenizer is not None, "tokenizer carregado")

    # 5. Dataset e DataLoader
    print("  [5] Criando dataset sintetico...")
    context_size = 64
    dataset = TextDataset(SAMPLE_TEXTS, tokenizer, context_size)
    loader  = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    check(len(dataset) > 0, f"dataset com {len(dataset)} amostras")
    batch_sample = next(iter(loader))
    check(batch_sample.shape[-1] == context_size,
          f"shape do batch: {tuple(batch_sample.shape)}")

    # 6. Student model (configuracao minima para teste rapido)
    print("  [6] Inicializando SOREModel v4.1 (config. minima)...")
    config = ModelConfig(
        vocab_size=len(tokenizer),
        context_size=context_size,
        embed_dim=128,
        num_layers=2,
        num_heads=4,
    )
    student = SOREModel_v4_1(config)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"       Parametros: {n_params:,}")
    check(student is not None, "student inicializado")

    # 7. DistillationTrainer
    print("  [7] Instanciando DistillationTrainer...")
    args = FakeArgs()
    trainer = DistillationTrainer(
        model=student,
        tokenizer=tokenizer,
        teacher_client=teacher,
        args=args,
        distillation_mode=mode,
    )
    check(trainer.distillation_mode == mode, f"modo='{trainer.distillation_mode}'")
    print(f"       Device: {trainer.device}")

    # 8. compute_loss (forward pass)
    print("  [8] Testando compute_loss (forward pass)...")
    batch = batch_sample.to(trainer.device)
    student.train()
    loss = trainer.compute_loss(student, batch)
    check(isinstance(loss, torch.Tensor), "loss e um Tensor")
    check(loss.ndim == 0, "loss e escalar")
    check(not torch.isnan(loss) and not torch.isinf(loss),
          f"loss valida: {loss.item():.4f}")

    # 9. Backward pass
    print("  [9] Testando backward pass...")
    loss.backward()
    grads_ok = any(p.grad is not None for p in student.parameters())
    check(grads_ok, "gradientes calculados com sucesso")
    trainer.optimizer.zero_grad()

    # 10. Treino de 1 epoca completa
    print("  [10] Rodando 1 epoca de treino...")
    trainer.train(loader, epochs=1)
    check(trainer.global_step > 0,
          f"global_step={trainer.global_step} apos treino")

    # 11. Validacao
    print("  [11] Rodando validate()...")
    val_loss = trainer.validate(loader)
    check(isinstance(val_loss, float), f"val_loss={val_loss:.4f}")
    check(val_loss > 0, "val_loss > 0")

    sep()
    print(f"  PASSOU — Modo '{mode}' funcionando corretamente!\n")
    return True


# -- Entry-point --------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Smoke-test da pipeline de distilacao do SOREModel")
    p.add_argument("--mode", type=str, default="sequence",
                   choices=["sequence", "hybrid", "all"],
                   help="Modo de destilacao a testar (default: sequence). "
                        "'logit' nao esta disponivel para Ollama (sem logits brutos).")
    p.add_argument("--ollama_url", type=str, default="http://localhost:11434")
    p.add_argument("--ollama_model", type=str, default="gemma4:latest")
    return p.parse_args()


def main():
    args = parse_args()

    sep("SOREModel - Smoke-test da Pipeline de Destilacao")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {'sim (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'nao (CPU)'}")
    print(f"  Teacher : {args.ollama_model} @ {args.ollama_url}")
    print(f"  Modo(s) : {args.mode}")

    # Ollama nao expoe logits brutos, entao o modo 'logit' puro
    # causaria RuntimeError intencional -- nao faz sentido testa-lo aqui.
    modes = ["sequence", "hybrid"] if args.mode == "all" else [args.mode]

    results = {}
    for mode in modes:
        try:
            results[mode] = run_test(mode, args.ollama_url, args.ollama_model)
        except AssertionError as e:
            results[mode] = False
            print(f"\n  FALHA no modo '{mode}': {e}\n")
        except Exception:
            results[mode] = False
            print(f"\n  ERRO INESPERADO no modo '{mode}':")
            traceback.print_exc()

    sep("Resumo")
    all_ok = True
    for mode, ok in results.items():
        status = "PASSOU" if ok else "FALHOU"
        print(f"  Modo '{mode}': {status}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  Todos os testes passaram! Pipeline de destilacao funcional.")
    else:
        print("  Alguns testes falharam. Verifique os erros acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()

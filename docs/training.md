
# Guia de Treinamento - SOREModel v4

Este documento detalha o pipeline de treinamento reformulado do SOREModel v4.

## Visão Geral

O pipeline suporta três estágios principais:
1.  **Pré-treino (Pretrain)**: Aprendizado autoregressivo em texto bruto.
2.  **Distilação (Distillation)**: Transferência de conhecimento de um modelo Teacher (OpenAI/Gemini/Local) para o SOREModel.
3.  **Instruction Tuning (SFT)**: Ajuste fino para seguir instruções.

## 1. Pré-treino Autoregressivo

Use `scripts/train.py` com `--stage pretrain`.

**Exemplo:**
```bash
python scripts/train.py \
    --stage pretrain \
    --dataset_name wikitext \
    --dataset_config wikitext-103-raw-v1 \
    --model_version v4 \
    --context_size 1024 \
    --batch_size 8 \
    --grad_accumulation_steps 4 \
    --epochs 10 \
    --use_amp \
    --output_dir ./checkpoints/pretrain_v4
```

## 2. Distilação Teacher-Student

Use `scripts/distill_sore.py`. Requer um dataset de prompts (texto).

**Exemplo (Teacher: Gemini):**
```bash
# Defina sua chave API antes
export GEMINI_API_KEY="sua_chave_aqui"

python scripts/distill_sore.py \
    --teacher_provider gemini \
    --teacher_model gemini-pro \
    --dataset_name "seu_dataset_de_prompts" \
    --distill_alpha 0.5 \
    --epochs 5 \
    --output_dir ./checkpoints/distill_gemini
```

**Exemplo (Teacher: OpenAI/Local vLLM):**
```bash
python scripts/distill_sore.py \
    --teacher_provider openai \
    --teacher_model gpt-3.5-turbo \
    --api_key "sk-..." \
    --epochs 5 \
    --output_dir ./checkpoints/distill_openai
```

## 3. Instruction Tuning (SFT)

Use `scripts/train.py` com `--stage sft`. Espera-se um dataset compatível com Hugging Face que contenha instruções.

**Exemplo:**
```bash
python scripts/train.py \
    --stage sft \
    --dataset_name "tatsu-lab/alpaca" \
    --resume_from_checkpoint ./checkpoints/pretrain_v4/final_model \
    --epochs 3 \
    --learning_rate 2e-5 \
    --output_dir ./checkpoints/sft_v4
```

## Parâmetros Importantes

- `--use_amp`: Ativa Mixed Precision (recomendado para GPU).
- `--lr_scheduler`: Define o scheduler ("cosine", "step").
- `--early_stopping_patience`: Número de épocas sem melhoria na validação antes de parar.

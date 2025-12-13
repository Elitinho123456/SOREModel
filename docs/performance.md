
# Performance e Otimização - SOREModel v4

Este documento descreve as técnicas de otimização implementadas no SOREModel v4.

## 1. Arquitetura Eficiente (v4)

- **Weight Tying**: A matriz de embeddings é reutilizada como a camada final de projeção (Head). Isso reduz significativamente o número de parâmetros.
- **RoPE (Rotary Positional Embeddings)**: Melhor representação posicional sem custo de parâmetros adicionais.
- **ALiBi (Attention with Linear Biases)**: Permite generalização para contextos maiores que o treino (extrapolação).
- **RMSNorm**: Normalização mais eficiente que LayerNorm padrão.

## 2. Treinamento Otimizado

- **AMP (Automatic Mixed Precision)**:
    - O treino roda em `float16` onde possível, mantendo precisão em `float32` apenas onde crítico.
    - Ative com `--use_amp`.
    - Reduz uso de VRAM e acelera o treino em GPUs Tensor Core.

- **DataLoaders**:
    - `num_workers` e `pin_memory` configuráveis.
    - Otimiza o fluxo de dados CPU -> GPU.

## 3. Inferência e Deploy

### Quantização Dinâmica (CPU)
Reduz o tamanho do modelo (ex: float32 -> int8 para pesos Lineares) para inferência rápida em CPU.

**Comando:**
```bash
python scripts/quantize_sore.py --checkpoint checkpoints/final_model/model.pt
```
Resultado: `model_quantized.pt` (aprox. 4x menor).

### Exportar para ONNX
Permite rodar o modelo em runtimes otimizados (ONNX Runtime, TensorRT).

**Comando:**
```bash
python scripts/export_onnx_sore.py --checkpoint checkpoints/final_model/model.pt --output sore_v4.onnx
```

## Comparativo de Tamanho (Estimado - 12 layers, 768 dim)

| Formato | Precisão | Tamanho Estimado | Uso |
| :--- | :--- | :--- | :--- |
| Checkpoint Padrão | FP32 | ~500 MB | Treino / Fine-tuning |
| Quantizado | INT8 | ~150 MB | Inferência CPU (Edge) |
| ONNX | FP32/FP16 | ~500 MB | Produção / Web |

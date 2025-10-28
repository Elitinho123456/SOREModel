# SOREModel

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**SOREModel** (Simple Open-Source Recurrent/Transformer Model) é uma biblioteca Python para criação e treinamento de modelos de linguagem baseados em Transformers. Implementado do zero para fins educacionais e de pesquisa.

## 🚀 Novidades na Versão 3.0

- 🆕 **SOREModel v3** com arquitetura avançada
- 🧠 Suporte a **ALiBi** (Attention with Linear Biases)
- 🔄 **RoPE** (Rotary Positional Embeddings)
- ⚡ **Otimizações de desempenho** com PyTorch
- 📊 Suporte a **Weights & Biases** para monitoramento

## ✨ Características

- 🧠 **Arquitetura Transformer avançada** implementada do zero
- 🔧 **Múltiplas versões**: v1 (simples), v2 (Transformer básico) e v3 (avançado com ALiBi e RoPE)
- 🎯 **Fácil de usar** com API limpa e intuitiva
- 📚 **Bem documentada** com exemplos completos
- 🚀 **Otimizada** para treinamento e inferência
- 🧪 **Testes abrangentes** incluídos

## 📦 Instalação

### Requisitos
- Python 3.8 ou superior
- PyTorch 1.9 ou superior
- NumPy

### Instalação via pip
```bash
# Dependências básicas
pip install -r requirements.txt

# Para treinamento (inclui suporte a GPU e W&B)
pip install -r requirements-train.txt
```

### Instalação do PyTorch
```bash
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU support (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Uso Rápido com v3

### 1. Treinamento do Modelo v3
```bash
# Treinar com configurações padrão
python scripts/train_sore_v3.py --use_wandb

# Treinar com parâmetros personalizados
python scripts/train_sore_v3.py \
  --dataset_name "seu_dataset" \
  --batch_size 32 \
  --context_size 1024 \
  --epochs 10 \
  --learning_rate 6e-4 \
  --output_dir ./checkpoints \
  --use_wandb
```

### 2. Carregando o Modelo v3
```python
from src.models.soreModel_v3 import SOREModel_v5, ModelConfig

# Configuração do modelo
config = ModelConfig(
    vocab_size=50000,       # Tamanho do vocabulário
    context_size=1024,      # Tamanho do contexto
    embed_dim=768,          # Dimensão dos embeddings
    num_heads=12,           # Número de cabeças de atenção
    num_layers=12,          # Número de camadas
    dropout=0.1,            # Dropout
    use_alibi=True,         # Usar ALiBi
    use_rmsnorm=True        # Usar RMSNorm
)

# Criar modelo
model = SOREModel_v5(config)

# Carregar pesos treinados (opcional)
checkpoint = torch.load('checkpoints/final_model/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### 3. Geração de Texto com v3
```python
# Configuração de geração
generation_config = {
    'max_length': 100,      # Comprimento máximo da geração
    'temperature': 0.7,     # Temperatura para amostragem
    'top_k': 50,            # Top-k sampling
    'top_p': 0.9,           # Nucleus sampling
    'repetition_penalty': 1.2  # Penalidade de repetição
}

# Gerar texto
input_text = "Olá, como vai"
generated = model.generate(
    input_text, 
    **generation_config
)
print(generated)
```

## 🏗️ Estrutura do Projeto Atualizada

```
SOREModel/
├── src/                        
│   ├── models/
│   │   ├── soreModel_v2.py     # Modelo Transformer v2
│   │   ├── soreModel_v3.py     # Modelo Transformer v3 (ALiBi + RoPE)
│   │   └── __init__.py
│   └── ...
├── scripts/
│   ├── train_sore_v3.py       # Script de treinamento v3
│   └── ...
├── checkpoints/               # Checkpoints dos modelos
├── requirements-train.txt     # Dependências para treinamento
└── ...
```

## 📊 Monitoramento com Weights & Biases

O script de treinamento inclui suporte nativo para o Weights & Biases. Para usar:

1. Instale o W&B:
```bash
pip install wandb
wandb login
```

2. Execute o treinamento com `--use_wandb`
3. Acompanhe as métricas em tempo real no [wandb.ai](https://wandb.ai)

## 🚀 Uso Rápido (Versões Anteriores)

### 1. Importação Básica
```python
import sys
sys.path.append('src')

from src import SOREModel_v2, criar_tokenizer_basico, Trainer, TextGenerator
```

### 2. Criando um Modelo
```python
# Configurações do modelo
vocab_size = 100
embed_dim = 32
context_size = 8
num_heads = 4
num_layers = 2

# Criar tokenizer
tokenizer = criar_tokenizer_basico()

# Criar modelo
modelo = SOREModel_v2(
    tamanho_vocab=tokenizer.get_vocab_size(),
    dim_embed=embed_dim,
    tamanho_contexto=context_size,
    num_heads=num_heads,
    num_layers=num_layers
)
```

### 3. Treinamento
```python
# Preparar dados de treinamento
textos = [
    "olá, mundo!",
    "como você está?",
    "estou bem, obrigado!",
    "que bom ouvir isso!"
]

# Criar trainer
trainer = Trainer(modelo, tokenizer)

# Treinar o modelo
trainer.treinar(
    textos=textos,
    contexto_tamanho=context_size,
    num_epocas=1000,
    batch_size=4,
    learning_rate=0.001
)
```

### 4. Geração de Texto
```python
# Criar gerador
gerador = TextGenerator(modelo, tokenizer)

# Gerar texto
contexto_inicial = "olá"
texto_gerado = gerador.gerar_texto(
    contexto_inicial,
    max_length=20,
    temperature=0.8
)

print(f"Texto gerado: {texto_gerado}")
```

## 📁 Estrutura do Projeto (Versões Anteriores)

```
SOREModel/
├── src/                        # Código fonte principal
│   ├── core/                   # Funcionalidades básicas
│   │   ├── tokenizer.py        # Tokenização de texto
│   │   └── __init__.py
│   ├── models/                 # Arquiteturas de modelo
│   │   ├── soreModel_v2.py     # Modelo Transformer v2
│   │   └── __init__.py
│   ├── training/               # Treinamento e ajuste fino
│   │   ├── trainer.py          # Classe Trainer
│   │   └── __init__.py
│   ├── generation/             # Geração de texto
│   │   ├── text_generator.py   # Classe TextGenerator
│   │   └── __init__.py
│   └── __init__.py             # Módulo principal
├── tests/                      # Testes
│   ├── test_soremodel_v2.py    # Testes completos
│   └── soreModel_v1_teste.py   # Testes da v1
├── examples/                   # Exemplos de uso
├── docs/                       # Documentação adicional
├── scripts/                    # Scripts utilitários
├── data/                       # Dados (se necessário)
├── requirements.txt            # Dependências
├── README.md                   # Este arquivo
└── LICENSE                     # Licença
```

## 🔧 API Reference

### SOREModel_v2
Modelo Transformer principal com Multi-Head Attention.

```python
class SOREModel_v2(nn.Module):
    def __init__(self, tamanho_vocab, dim_embed, tamanho_contexto, num_heads, num_layers):
        # Parâmetros:
        # - tamanho_vocab: Tamanho do vocabulário
        # - dim_embed: Dimensão dos embeddings
        # - tamanho_contexto: Tamanho máximo do contexto
        # - num_heads: Número de cabeças de atenção
        # - num_layers: Número de blocos Transformer
```

### Tokenizer
Classe para tokenização de caracteres.

```python
tokenizer = criar_tokenizer_basico()

# Codificar texto
indices = tokenizer.codificar("olá, mundo!")

# Decodificar índices
texto = tokenizer.decodificar(indices)

# Tamanho do vocabulário
vocab_size = tokenizer.get_vocab_size()
```

### Trainer
Classe para treinamento e ajuste fino.

```python
trainer = Trainer(modelo, tokenizer)

# Treinamento básico
trainer.treinar(textos, contexto_tamanho, num_epocas=1000)

# Ajuste fino
trainer.ajustar_fino(novos_textos, contexto_tamanho, num_epocas=500)

# Salvar/carregar modelo
trainer.salvar_modelo("meu_modelo.pth")
trainer.carregar_modelo("meu_modelo.pth")
```

### TextGenerator
Classe para geração de texto.

```python
gerador = TextGenerator(modelo, tokenizer)

# Geração básica
texto = gerador.gerar_texto("olá", max_length=20)

# Geração com parâmetros avançados
texto = gerador.gerar_texto(
    "olá",
    max_length=50,
    temperature=0.8,    # Criatividade
    top_k=40,          # Top-k sampling
    top_p=0.9          # Nucleus sampling
)

# Beam search
texto = gerador.gerar_texto_beam_search("olá", beam_width=3)

# Completar texto
texto = gerador.completar_texto("texto-incompleto", max_completar=10)
```

## 🧪 Executando Testes

```bash
# Executar todos os testes
python tests/test_soremodel_v2.py

# Executar testes específicos
python -m pytest tests/
```

## 📖 Exemplos

### Exemplo Básico
```python
import sys
sys.path.append('src')

from src import SOREModel_v2, criar_tokenizer_basico, Trainer, TextGenerator

# 1. Preparar dados
tokenizer = criar_tokenizer_basico()
textos = ["olá", "oi", "tchau", "adeus"]

# 2. Criar modelo
modelo = SOREModel_v2(
    tamanho_vocab=tokenizer.get_vocab_size(),
    dim_embed=32,
    tamanho_contexto=8,
    num_heads=4,
    num_layers=2
)

# 3. Treinar
trainer = Trainer(modelo, tokenizer)
trainer.treinar(textos, contexto_tamanho=3, num_epocas=200)

# 4. Gerar texto
gerador = TextGenerator(modelo, tokenizer)
resultado = gerador.gerar_texto("olá", max_length=10)
print(resultado)
```

### Exemplo Avançado com Ajuste Fino
```python
# Treinamento inicial
trainer.treinar(textos_basicos, contexto_tamanho=8, num_epocas=1000)

# Ajuste fino com novos dados
novos_textos = ["novos dados", "para aprender", "padrões diferentes"]
trainer.ajustar_fino(novos_textos, contexto_tamanho=8, num_epocas=500)

# Geração com diferentes temperaturas
for temp in [0.5, 0.8, 1.2]:
    texto = gerador.gerar_texto("início", max_length=20, temperature=temp)
    print(f"Temperatura {temp}: {texto}")
```

## 🎯 Comparação entre Versões

| Característica | SOREModel v1 | SOREModel v2 |
|----------------|--------------|--------------|
| Arquitetura | Linear simples | Transformer |
| Multi-Head Attention | ❌ | ✅ |
| Position Embedding | ❌ | ✅ |
| Layer Normalization | ❌ | ✅ |
| Residual Connections | ❌ | ✅ |
| Complexidade | Baixa | Alta |
| Qualidade de Geração | Básica | Avançada |

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor, leia as diretrizes de contribuição:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 📧 Contato

- **Projeto**: SOREModel
- **Versão**: 2.0.0 BETA
- **Autor**: SOREModel Team
- **Issues**: [GitHub Issues](https://github.com/Elitinho123456/SOREModel/issues)

## 🙏 Agradecimentos

- PyTorch team pela excelente biblioteca
- Comunidade open-source por inspiração e suporte
- Contribuidores e usuários do projeto

---

<div align="center">

**Feito com ❤️ pela comunidade open-source**

[⭐ Star](https://github.com/Elitinho123456/SOREModel) | [🐛 Reportar Bug](https://github.com/Elitinho123456/SOREModel/issues) | [💬 Discussões](https://github.com/Elitinho123456/SOREModel/discussions)

</div>

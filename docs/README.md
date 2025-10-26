# SOREModel v2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**SOREModel** (Simple Open-Source Recurrent/Transformer Model) é uma biblioteca Python para criação e treinamento de modelos de linguagem baseados em Transformers. Implementado do zero para fins educacionais e de pesquisa.

## ✨ Características

- 🧠 **Arquitetura Transformer completa** implementada do zero
- 🔧 **Múltiplas versões**: v1 (simples) e v2 (Transformer avançado)
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
pip install -r requirements.txt
```

### Instalação do PyTorch
```bash
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU support (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Uso Rápido

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

## 📁 Estrutura do Projeto

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

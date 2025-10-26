# 📊 Dados da Wikipedia para SOREModel

Este diretório contém dados e utilitários para coletar e processar artigos da Wikipedia em português para treinamento do SOREModel.

## 📁 Arquivos

### `wikipedia_articles_pt.json`
Arquivo principal com **120 links** de artigos da Wikipedia em português, organizados por categorias:

- **Ciência e Tecnologia** (20 artigos): IA, Machine Learning, Robótica, Programação, etc.
- **História** (20 artigos): Brasil, Portugal, Guerras Mundiais, Revoluções, etc.
- **Geografia** (20 artigos): Brasil, Portugal, países da Europa e América, etc.
- **Literatura** (20 artigos): Autores brasileiros, portugueses e internacionais
- **Artes** (20 artigos): Pintura, Música, Teatro, Cinema, movimentos artísticos
- **Esportes** (20 artigos): Futebol, Olimpíadas, esportes diversos
- **Política** (20 artigos): Sistemas políticos, organizações internacionais
- **Biologia** (20 artigos): Evolução, genética, ecossistemas brasileiros
- **Física** (20 artigos): Mecânica, relatividade, energia, termodinâmica
- **Matemática** (20 artigos): Áreas da matemática pura e aplicada

### `wikipedia_utils.py`
Utilitário completo para:
- ✅ Carregar dados do JSON
- ✅ Extrair conteúdo de artigos via web scraping
- ✅ Processar múltiplos artigos
- ✅ Salvar conteúdo extraído
- ✅ Gerenciar categorias e URLs

## 🚀 Como Usar

### 1. Carregar Dados
```python
from data.wikipedia_utils import WikipediaDataManager

# Inicializar gerenciador
data_manager = WikipediaDataManager()

# Carregar dados
data = data_manager.load_articles_data()

# Ver categorias disponíveis
categories = data_manager.get_categories()
print(f"Categorias: {categories}")

# Pegar URLs de uma categoria
urls = data_manager.get_urls_by_category('ciencia_tecnologia')
print(f"Artigos de ciência: {len(urls)}")
```

### 2. Extrair Conteúdo
```python
# Extrair um artigo específico
url = "https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial"
content = data_manager.scrape_article_content(url, max_paragraphs=10)

if content:
    print(f"Conteúdo extraído: {len(content)} caracteres")
    print(content[:200] + "...")
```

### 3. Processar Múltiplos Artigos
```python
# Pegar todas as URLs
all_urls = data_manager.get_all_urls()

# Extrair conteúdo de vários artigos
results = data_manager.scrape_multiple_articles(
    urls=all_urls[:5],  # Primeiros 5 artigos
    max_paragraphs=20,   # Máximo 20 parágrafos por artigo
    delay=2.0           # 2 segundos entre requests
)

# Salvar resultados
data_manager.save_scraped_content(results, 'data/artigos_processados.json')
```

## 📋 Exemplo de Uso com SOREModel

```python
import sys
sys.path.append('src')

from data.wikipedia_utils import WikipediaDataManager
from src import Trainer, SOREModel_v2, criar_tokenizer_basico

# 1. Carregar dados da Wikipedia
data_manager = WikipediaDataManager()
urls = data_manager.get_all_urls()

# 2. Extrair conteúdo (exemplo com 10 artigos)
results = data_manager.scrape_multiple_articles(urls[:10], max_paragraphs=15)

# 3. Preparar textos para treinamento
textos_treinamento = [content for url, content in results]

# 4. Treinar modelo
tokenizer = criar_tokenizer_basico()
modelo = SOREModel_v2(
    tamanho_vocab=tokenizer.get_vocab_size(),
    dim_embed=64,
    tamanho_contexto=16,
    num_heads=8,
    num_layers=4
)

trainer = Trainer(modelo, tokenizer)
trainer.treinar(
    textos=textos_treinamento,
    contexto_tamanho=8,
    num_epocas=1000,
    batch_size=4
)
```

## 📊 Estatísticas

- **Total de links**: 120
- **Categorias**: 10
- **Artigos por categoria**: 20
- **Idioma**: Português (Brasil)
- **Fontes**: Wikipedia (pt.wikipedia.org)
- **Temas**: Diversos (ciência, história, geografia, artes, etc.)

## ⚠️ Avisos Importantes

1. **Respeite os termos de uso** da Wikipedia
2. **Use delays apropriados** entre requests (recomendado: 1-2 segundos)
3. **Limite a quantidade** de artigos processados por vez
4. **Verifique a conectividade** antes de executar scraping em larga escala
5. **Considere usar APIs oficiais** quando disponíveis

## 🔧 Funções Disponíveis

### WikipediaDataManager
- `load_articles_data()`: Carrega dados do JSON
- `get_all_urls()`: Retorna todas as URLs
- `get_urls_by_category(category)`: URLs por categoria
- `get_categories()`: Lista de categorias
- `scrape_article_content(url, max_paragraphs)`: Extrai um artigo
- `scrape_multiple_articles(urls, max_paragraphs, delay)`: Extrai múltiplos
- `save_scraped_content(results, output_file)`: Salva resultados

## 📈 Expansão Futura

Para expandir a base de dados:

1. **Adicionar mais categorias** no JSON
2. **Incluir mais artigos** por categoria
3. **Diversificar fontes** (além da Wikipedia)
4. **Implementar cache** para evitar re-download
5. **Adicionar limpeza de texto** mais sofisticada

## 🤝 Contribuição

Para contribuir com mais links:

1. Adicione URLs no arquivo `wikipedia_articles_pt.json`
2. Mantenha a estrutura por categorias
3. Teste os links antes de submeter
4. Documente novas categorias no README

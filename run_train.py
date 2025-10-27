import sys
import json
import os

sys.path.append('src') 

from src import Trainer, SOREModel_v2, Tokenizer

caminho_dados = os.path.join('data', 'corpus_completo.json')
print(f"Carregando dados do corpus local: {caminho_dados}...")

try:
    with open(caminho_dados, 'r', encoding='utf-8') as f:
        dados_salvos = json.load(f)
except FileNotFoundError:
    print(f"Erro: Arquivo '{caminho_dados}' não encontrado.")
    print("Você precisa executar o script 'coletar_dados.py' primeiro.")
    sys.exit(1)

textos_treinamento = [artigo['content'] for artigo in dados_salvos['articles']]
print(f"Dados carregados: {len(textos_treinamento)} artigos.")


print("Configurando modelo e tokenizador...")
tokenizador = Tokenizer()

# Usando as configurações recomendadas (contexto maior)
CONTEXTO_TAMANHO = 128
BATCH_SIZE = 4 

modelo = SOREModel_v2(
    tamanho_vocab=tokenizador.get_vocab_size(),
    dim_embed=128,
    tamanho_contexto=CONTEXTO_TAMANHO,
    num_heads=8,
    num_layers=4
)

trainer = Trainer(modelo, tokenizador)

print(f"Iniciando treinamento com Contexto={CONTEXTO_TAMANHO} e Batch={BATCH_SIZE}")
try:
    trainer.treinar(
        textos=textos_treinamento,
        contexto_tamanho=CONTEXTO_TAMANHO, # Importante: ser igual ao do modelo
        num_epocas=1000,
        batch_size=BATCH_SIZE,
        learning_rate=0.001
    )
except KeyboardInterrupt:
    print("\nTreinamento interrompido pelo usuário.")
except Exception as e:
    print(f"\nErro durante o treinamento: {e}")
finally:
    print("Loop de treinamento finalizado.")

    # Salvar o modelo em uma pasta chamada 'modelos_checkpoints' dentro de  data_path/modelos_checkpoints
    os.makedirs(os.path.join('src', 'models', 'modelos_checkpoints'), exist_ok=True)
    trainer.salvar_modelo(os.path.join('src', 'models', 'modelos_checkpoints', 'modelo_checkpoint_final.pth'))
    print("Modelo final salvo.")
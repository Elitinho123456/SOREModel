import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from src import Trainer, SOREModel_v2, Tokenizer, TextGenerator
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print("Verifique se a pasta 'src' está acessível e contém os arquivos .py necessários.")
    sys.exit(1)

# --- CONFIGURAÇÃO DE CARREGAMENTO ---
# (Ajuste estes parâmetros para corresponder ao modelo que você TREINOU)

CONTEXTO_TAMANHO = 128
DIM_EMBED = 128
NUM_HEADS = 8
NUM_LAYERS = 4 
CHECKPOINT_PATH = 'src/models/modelos_checkpoints/modelo_checkpoint_final.pth'


print("Carregando SOREModel v2 (BPE)...")

try:

    tokenizador = Tokenizer() 
    tamanho_vocabulario = tokenizador.get_vocab_size()
    print(f"Tokenizer BPE carregado (Vocabulário: {tamanho_vocabulario} tokens)")

    modelo = SOREModel_v2(
        tamanho_vocab=tamanho_vocabulario,
        dim_embed=DIM_EMBED,
        tamanho_contexto=CONTEXTO_TAMANHO,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS                                           # IMPORTANTE: Deve ser igual ao modelo salvo!
    )

    trainer = Trainer(modelo, tokenizador)
    trainer.carregar_modelo(CHECKPOINT_PATH)
    print(f"Checkpoint '{CHECKPOINT_PATH}' carregado com sucesso.")

    gerador = TextGenerator(trainer.modelo, tokenizador)

except FileNotFoundError as e:
    print(f"--- ERRO CRÍTICO ---")
    print(f"Não foi possível encontrar um arquivo necessário: {e.filename}")
    print("Verifique se você treinou o tokenizador BPE e se o checkpoint do modelo existe.")
    sys.exit(1)
except Exception as e:
    print(f"Ocorreu um erro inesperado durante o carregamento: {e}")
    sys.exit(1)

# --- 3. LOOP DE CHAT INTERATIVO ---
print("\n--- CHAT SOREMODEL v2 (BPE) ---")
print("Modelo carregado. Pronto para conversar.")
print("Comandos:")
print("  /beam [prompt]  -> Usa Beam Search (para coerência)")
print("  /sample [prompt] -> Usa Amostragem (padrão, para criatividade)")
print("  /quit           -> Sair")

while True:
    prompt = input("\nVocê: ")

    if prompt.lower() == '/quit':
        print("Até logo!")
        break
    
    if not prompt:                                                      # Ignorar entrada vazia
        continue

    # --- Lógica do Roteador (Opção de Escolha) ---
    clean_prompt = prompt
    metodo = "sample"                                                   # Padrão

    if prompt.startswith('/beam '):
        metodo = "beam"
        clean_prompt = prompt.removeprefix('/beam ').strip()
    elif prompt.startswith('/sample '):
        metodo = "sample"
        clean_prompt = prompt.removeprefix('/sample ').strip()
    # --- Fim do Roteador ---

    print("SOREModel:", end="", flush=True)                             # Para a resposta aparecer na mesma linha

    if metodo == "beam":
        resposta = gerador.gerar_texto_beam_search(
            clean_prompt,
            max_length=50,                                              # Mais curto, pois é computacionalmente caro
            beam_width=3
        )
    else:
        resposta = gerador.gerar_texto(
            clean_prompt,
            max_length=100,                                             # Pode ser mais longo
            temperature=0.8,
            top_p=0.9
        )
        
                                                                        # Limpar a resposta (o gerador retorna o prompt + a resposta)
    if resposta.startswith(clean_prompt):
         resposta_limpa = resposta.removeprefix(clean_prompt)
    else:
         resposta_limpa = resposta                                      # Fallback
         
    print(resposta_limpa)
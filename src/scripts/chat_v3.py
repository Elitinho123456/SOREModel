"""
chat_v3.py
Script para interagir com um modelo SOREModel v3 treinado.
"""
import os
import sys
import torch
from pathlib import Path

# Adiciona o diretório raiz ao path para importação correta dos módulos
caminho_projeto = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(caminho_projeto))

try:
    from src.models.soreModel_v3 import SOREModel_v3, ModelConfig
    from src.core.tokenizer_pipeline import load_tokenizer
    from src.generation.text_generator import TextGenerator
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print(f"Verifique se o caminho do projeto '{caminho_projeto}' está correto e os módulos existem.")
    sys.exit(1)

# --- Configurações ---
CHECKPOINT_PATH = caminho_projeto / 'checkpoints_v3' / 'checkpoint_passo_14000' / 'modelo.pt'
TOKENIZER_DIR = caminho_projeto / 'tokenizer'

def carregar_modelo_e_tokenizer(checkpoint_path, tokenizer_dir):
    """Carrega o modelo, o tokenizador e a configuração a partir de um checkpoint."""
    print("Iniciando o carregamento do modelo e tokenizador...")
    
    if not checkpoint_path.exists():
        print(f"ERRO: Arquivo de checkpoint não encontrado em {checkpoint_path}")
        return None, None, None

    # Carregar o checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("Checkpoint carregado com sucesso.")
    except Exception as e:
        print(f"ERRO ao carregar o arquivo de checkpoint: {e}")
        return None, None, None

    # Carregar configuração do modelo a partir do checkpoint
    if 'config' not in checkpoint:
        print("ERRO: O checkpoint não contém a chave 'config' com os parâmetros do modelo.")
        return None, None, None
    
    model_args = checkpoint['config']
    config = ModelConfig(**model_args)
    
    print("\nConfiguração do Modelo (carregada do checkpoint):")
    for key, value in model_args.items():
        print(f"- {key}: {value}")

    # Inicializar o modelo com a configuração carregada
    modelo = SOREModel_v3(config)
    
    # Carregar os pesos do modelo
    if 'modelo_state_dict' in checkpoint:
        modelo.load_state_dict(checkpoint['modelo_state_dict'])
        print("\nPesos do modelo (state_dict) carregados.")
    else:
        print("ERRO: O checkpoint não contém a chave 'modelo_state_dict'.")
        return None, None, None

    # Carregar o tokenizador
    try:
        tokenizer = load_tokenizer(str(tokenizer_dir))
        print(f"Tokenizador carregado de '{tokenizer_dir}'. Tamanho do vocabulário: {tokenizer.get_vocab_size()}")
    except FileNotFoundError as e:
        print(f"ERRO: {e}")
        return None, None, None

    # Configurar dispositivo e modo de avaliação
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo.to(dispositivo)
    modelo.eval()
    print(f"Modelo movido para o dispositivo: {dispositivo}")

    return modelo, tokenizer, dispositivo

def main():
    """Função principal para executar o chat interativo."""
    modelo, tokenizer, dispositivo = carregar_modelo_e_tokenizer(CHECKPOINT_PATH, TOKENIZER_DIR)

    if not all([modelo, tokenizer, dispositivo]):
        print("\nFalha ao carregar os componentes necessários. Encerrando o script.")
        return

    # Inicializar o gerador de texto
    gerador = TextGenerator(modelo, tokenizer, dispositivo)

    print("\n--- CHAT SOREMODEL v3 ---")
    print("Modelo carregado. Pronto para gerar texto.")
    print("Comandos:")
    print("  /beam [prompt]  -> Usa Beam Search (para coerência)")
    print("  /sample [prompt] -> Usa Amostragem (padrão, para criatividade)")
    print("  /quit           -> Sair")

    metodo = "sample"

    while True:
        prompt = input("\nVocê: ")
        if not prompt:
            continue

        clean_prompt = prompt.strip()

        if clean_prompt.lower() == '/quit':
            print("Até logo!")
            break
        
        if clean_prompt.lower().startswith('/beam'):
            metodo = "beam"
            clean_prompt = clean_prompt[5:].strip()
            print(f"(Modo: Beam Search, Prompt: '{clean_prompt}')")
        elif clean_prompt.lower().startswith('/sample'):
            metodo = "sample"
            clean_prompt = clean_prompt[7:].strip()
            print(f"(Modo: Amostragem, Prompt: '{clean_prompt}')")

        if not clean_prompt:
            print("Por favor, insira um texto após o comando.")
            continue

        print("SOREModel v3: ", end="", flush=True)
        
        try:
            if metodo == "beam":
                resposta = gerador.gerar_texto_beam_search(
                    contexto_inicial=clean_prompt,
                    max_length=100,
                    beam_width=3
                )
            else: # sample
                resposta = gerador.gerar_texto(
                    contexto_inicial=clean_prompt,
                    max_length=150,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9
                )
            print(resposta.replace(clean_prompt, "").strip())

        except Exception as e:
            print(f"\nERRO durante a geração de texto: {e}")

if __name__ == "__main__":
    main()

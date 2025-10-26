import json
import os
import sys

def criar_vocabulario_de_caracteres():
    """
    Lê o corpus de dados e gera um vocabulário de caracteres únicos.
    """
    
    caminho_dados = os.path.join('data', 'corpus_completo.json')
    print(f"Iniciando a leitura do corpus em: {caminho_dados}...")

    if not os.path.exists(caminho_dados):
        print(f"Erro: Arquivo '{caminho_dados}' não encontrado.")
        print("Por favor, execute o script 'coletar_dados.py' primeiro.")
        sys.exit(1)

    caracteres_unicos = set()

    try:
        with open(caminho_dados, 'r', encoding='utf-8') as f:
            dados_salvos = json.load(f)

        # Acessa a lista de artigos
        lista_artigos = dados_salvos.get('articles', [])
        if not lista_artigos:
            print("Aviso: A chave 'articles' não foi encontrada ou está vazia no JSON.")
            return

        print(f"Processando {len(lista_artigos)} artigos...")

        # Itera por cada artigo e adiciona seu conteúdo ao set
        for i, artigo in enumerate(lista_artigos):
            conteudo = artigo.get('content')
            if conteudo:
                # 'update' adiciona todos os caracteres da string ao set
                caracteres_unicos.update(conteudo)

        print("Processamento concluído.")

        vocabulario_lista = sorted(list(caracteres_unicos))

    except json.JSONDecodeError:
        print(f"Erro: Falha ao decodificar o JSON em '{caminho_dados}'.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

    return vocabulario_lista
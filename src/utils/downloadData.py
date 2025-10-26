import sys
import os
import time

# Adiciona a pasta 'src' ao path para encontrar o 'data.wikipedia_utils'
sys.path.append('src') 

try:
    from data.wikipedia_utils import WikipediaDataManager
except ImportError:
    print("Erro: Não foi possível encontrar o 'WikipediaDataManager'.")
    print("Certifique-se que a pasta 'src' está no local correto.")
    sys.exit(1)

print("Iniciando coleta de dados...")

data_manager = WikipediaDataManager()

urls = data_manager.get_all_urls()
print(f"Encontrados {len(urls)} URLs para coletar.")

os.makedirs('data', exist_ok=True)

caminho_saida = os.path.join('data', 'corpus_completo.json')

start_time = time.time()
results = data_manager.scrape_multiple_articles(urls, delay=1.0) # 
data_manager.save_scraped_content(results, caminho_saida) 

end_time = time.time()
print("\n--- Coleta Concluída ---")
print(f"Tempo total: {(end_time - start_time):.2f} segundos")
print(f"Dados salvos com sucesso em: {caminho_saida}")
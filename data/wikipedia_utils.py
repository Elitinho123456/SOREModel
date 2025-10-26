"""
Utilitário para carregar e processar dados da Wikipedia
Funcionalidades para scraping e processamento de artigos
"""

import json
import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple


class WikipediaDataManager:
    """Gerenciador para dados da Wikipedia em português"""

    def __init__(self, data_file: str = None):
        """
        Inicializa o gerenciador de dados

        Args:
            data_file (str): Caminho para o arquivo JSON com links da Wikipedia
        """
        if data_file is None:
            # Usa o arquivo padrão na pasta data
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_file = os.path.join(project_root, 'data', 'wikipedia_articles_pt.json')

        self.data_file = data_file
        self.articles_data = None

    def load_articles_data(self) -> Dict:
        """
        Carrega os dados dos artigos da Wikipedia

        Returns:
            Dict: Dados carregados do JSON
        """
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.articles_data = json.load(f)
            print(f"✅ Dados carregados: {self.articles_data['metadata']['total_links']} links encontrados")
            return self.articles_data
        except FileNotFoundError:
            print(f"❌ Arquivo não encontrado: {self.data_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Erro ao decodificar JSON: {e}")
            return {}

    def get_all_urls(self) -> List[str]:
        """
        Retorna todas as URLs de artigos

        Returns:
            List[str]: Lista com todas as URLs
        """
        if not self.articles_data:
            self.load_articles_data()

        all_urls = []
        for category, urls in self.articles_data.get('articles', {}).items():
            all_urls.extend(urls)

        return all_urls

    def get_urls_by_category(self, category: str) -> List[str]:
        """
        Retorna URLs de uma categoria específica

        Args:
            category (str): Nome da categoria

        Returns:
            List[str]: Lista de URLs da categoria
        """
        if not self.articles_data:
            self.load_articles_data()

        return self.articles_data.get('articles', {}).get(category, [])

    def get_categories(self) -> List[str]:
        """
        Retorna todas as categorias disponíveis

        Returns:
            List[str]: Lista de categorias
        """
        if not self.articles_data:
            self.load_articles_data()

        return self.articles_data.get('metadata', {}).get('categories', [])

    def scrape_article_content(self, url: str, max_paragraphs: int = None) -> Optional[str]:
        """
        Extrai o conteúdo de um artigo da Wikipedia

        Args:
            url (str): URL do artigo
            max_paragraphs (int): Número máximo de parágrafos (opcional)

        Returns:
            Optional[str]: Conteúdo do artigo ou None se erro
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove elementos indesejados
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()

            # Extrai parágrafos do conteúdo principal
            content_paragraphs = []

            # Tenta encontrar o conteúdo principal
            content_selectors = [
                'div.mw-content-ltr',
                'div.mw-parser-output',
                'div#content',
                'article'
            ]

            main_content = None
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            if main_content:
                paragraphs = main_content.find_all('p')
            else:
                paragraphs = soup.find_all('p')

            for p in paragraphs:
                text = p.get_text().strip()
                if text and len(text) > 20:  # Filtra parágrafos muito curtos
                    content_paragraphs.append(text)

            if max_paragraphs:
                content_paragraphs = content_paragraphs[:max_paragraphs]

            return '\n\n'.join(content_paragraphs)

        except requests.RequestException as e:
            print(f"❌ Erro ao acessar {url}: {e}")
            return None
        except Exception as e:
            print(f"❌ Erro ao processar {url}: {e}")
            return None

    def scrape_multiple_articles(self, urls: List[str], max_paragraphs: int = None,
                                delay: float = 1.0) -> List[Tuple[str, str]]:
        """
        Extrai conteúdo de múltiplos artigos

        Args:
            urls (List[str]): Lista de URLs
            max_paragraphs (int): Máximo de parágrafos por artigo
            delay (float): Delay entre requests em segundos

        Returns:
            List[Tuple[str, str]]: Lista de tuplas (url, conteúdo)
        """
        import time

        results = []
        successful = 0
        failed = 0

        print(f"📚 Iniciando scraping de {len(urls)} artigos...")

        for i, url in enumerate(urls, 1):
            print(f"  [{i}/{len(urls)}] Processando: {url}")

            content = self.scrape_article_content(url, max_paragraphs)

            if content:
                results.append((url, content))
                successful += 1
                print(f"    ✅ Sucesso ({len(content)} caracteres)")
            else:
                failed += 1
                print("    ❌ Falha")

            if i < len(urls) and delay > 0:
                time.sleep(delay)

        print(f"\n📊 Resumo: {successful} sucessos, {failed} falhas")
        return results

    def save_scraped_content(self, results: List[Tuple[str, str]], output_file: str):
        """
        Salva o conteúdo extraído em um arquivo

        Args:
            results (List[Tuple[str, str]]): Resultados do scraping
            output_file (str): Arquivo de saída
        """
        output_data = {
            'metadata': {
                'total_articles': len(results),
                'source': 'wikipedia_pt',
                'created': '2025-10-26'
            },
            'articles': []
        }

        total_chars = 0
        for url, content in results:
            output_data['articles'].append({
                'url': url,
                'content': content,
                'char_count': len(content)
            })
            total_chars += len(content)

        # Salvar como JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Dados salvos em: {output_file}")
        print(f"📊 Total: {len(results)} artigos, {total_chars} caracteres")


def main():
    """Função principal para teste"""
    data_manager = WikipediaDataManager()

    # Carregar dados
    data = data_manager.load_articles_data()
    if not data:
        return

    # Mostrar estatísticas
    categories = data_manager.get_categories()
    print(f"\n📊 Categorias disponíveis: {len(categories)}")
    for category in categories:
        urls = data_manager.get_urls_by_category(category)
        print(f"  - {category}: {len(urls)} artigos")

    # Exemplo de scraping
    print("\n🔍 Exemplo de scraping (primeiros 3 artigos):")
    all_urls = data_manager.get_all_urls()
    sample_urls = all_urls[:3]

    results = data_manager.scrape_multiple_articles(sample_urls, max_paragraphs=5, delay=2)

    if results:
        print("\n📝 Amostra de conteúdo extraído:")
        for url, content in results[:1]:  # Mostra apenas o primeiro
            print(f"\nURL: {url}")
            print(f"Conteúdo (primeiros 200 caracteres): {content[:200]}...")

        # Salvar exemplo
        data_manager.save_scraped_content(results, 'data/artigos_exemplo.json')


if __name__ == "__main__":
    main()

import requests
from bs4 import BeautifulSoup

url = 'https://pt.wikipedia.org/wiki/Brasil'

def pesquisar_conteudo(url):

    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    resposta = requests.get(url, headers=header)

    soup = BeautifulSoup(resposta.text, 'html.parser')

    conteudo = []
    
    for paragrafo in soup.find_all('p'):
        conteudo.append(paragrafo.get_text())

    return '\n'.join(conteudo)
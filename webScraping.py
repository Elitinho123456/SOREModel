import requests
from bs4 import BeautifulSoup

url = 'https://pt.wikipedia.org/wiki/Brasil'

header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

resposta = requests.get(url, headers=header)
print(resposta.status_code)

soup = BeautifulSoup(resposta.text, 'html.parser')

conteudo = []

for paragrafo in soup.find_all('p'):
    conteudo.append(paragrafo.get_text())

print(len(conteudo))

conteudo_completo = '\n'.join(conteudo)
print(conteudo_completo)

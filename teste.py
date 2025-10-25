import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Pilar 1: Tokenização ---
modeloConhecimento = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','Á','À','Ã','Â','Ä','É','È','Ê','Ë','Í','Ì','Î','Ï','Ó','Ò','Õ','Ô','Ö','Ú','Ù','Û','Ü','Ç','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','á','à','ã','â','ä','é','è','ê','ë','í','ì','î','ï','ó','ò','õ','ô','ö','ú','ù','û','ü','ç',' ',',','!','.','?']
vocabulario = sorted(list(modeloConhecimento))
tamanho_vocabulario = len(vocabulario)

stoi = {char: i for i, char in enumerate(vocabulario)}
itos = {i: char for i, char in enumerate(vocabulario)}

def codificar(texto):
    return [stoi[char] for char in texto]

def decodificar(indices):
    return ''.join([itos[i] for i in indices])

# --- Pilar 2: Preparação dos Dados ---
texto = "olá, mundo!"
textoN = "oi, tchau!"
tamanho_de_contexto = 3
exemplos_x = []
exemplos_xN = []
exemplos_y = []
exemplos_yN = []

dados_codificados = codificar(texto)
for i in range(len(dados_codificados) - tamanho_de_contexto): 
    x = dados_codificados[i : i + tamanho_de_contexto]
    y = dados_codificados[i + tamanho_de_contexto]
    exemplos_x.append(x)
    exemplos_y.append(y)

dados_codificadosN = codificar(textoN)
for i in range(len(dados_codificadosN) - tamanho_de_contexto): 
    x = dados_codificadosN[i : i + tamanho_de_contexto]
    y = dados_codificadosN[i + tamanho_de_contexto]
    exemplos_xN.append(x)
    exemplos_yN.append(y)

X_tensor = torch.tensor(exemplos_x, dtype=torch.long)
Y_tensor = torch.tensor(exemplos_y, dtype=torch.long)

X_tensorN = torch.tensor(exemplos_xN, dtype=torch.long)
Y_tensorN = torch.tensor(exemplos_yN, dtype=torch.long)

X_tensor_final = torch.cat((X_tensor, X_tensorN), dim=0)
Y_tensor_final = torch.cat((Y_tensor, Y_tensorN), dim=0)


class atentionHead(nn.Module):
    def __init__(self, dim_entrada, dim_saida_qkv):
        super(atentionHead, self).__init__()
        self.camada_query = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)
        self.camada_key = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)
        self.camada_value = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)
        
    def forward(self, x):

        B, T, C = x.shape  # Batch size, Tamanho da sequência, Dimensão de entrada

        Q = self.camada_query(x)
        K = self.camada_key(x)
        V = self.camada_value(x)

        dim_k = K.size(-1)
        pontuacoes = torch.matmul(Q, K.transpose(-2, -1)) / (dim_k ** 0.5)

        mascara = torch.tril(torch.ones(T, T)).to(x.device)  # Máscara triangular inferior
        pontuacoes_mascaradas = pontuacoes.masked_fill(mascara == 0, float('-inf'))

        pesos_atencao = F.softmax(pontuacoes_mascaradas, dim=-1)

        saida = torch.matmul(pesos_atencao, V)

        return saida
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_entrada, dim_saida_qkv):
        super(MultiHeadAttention, self).__init__()
        
        self.heads = nn.ModuleList([atentionHead(dim_entrada, dim_saida_qkv) for _ in range(num_heads)])

        self.projecao = nn.Linear(in_features=num_heads * dim_saida_qkv, out_features=dim_entrada)

    def forward(self, x):
        saidas_cabecas = [head(x) for head in self.heads]
        saida_concatenada = torch.cat(saidas_cabecas, dim=-1)

        saida_final = self.projecao(saida_concatenada)

        return saida_final
    
class Block(nn.Module):
    def __init__(self, num_heads, dim_entrada, dim_saida_qkv):
        super(Block, self).__init__()
        
        # --- Self-Attention ---
        self.atencao = MultiHeadAttention(num_heads, dim_entrada, dim_saida_qkv)
        self.norm1 = nn.LayerNorm(dim_entrada)

        # --- Feed-Forward ---
        dim_ffn = 4 * dim_entrada

        self.ffn = nn.Sequential(
            nn.Linear(in_features=dim_entrada, out_features=dim_ffn),
            nn.ReLU(),
            nn.Linear(in_features=dim_ffn, out_features=dim_entrada)
        )
        self.norm2 = nn.LayerNorm(dim_entrada)

    def forward(self, x):

        saida_atencao = self.atencao(x)
        x_normalizado1 = self.norm1(x + saida_atencao)

        saida_ffn = self.ffn(x_normalizado1)
        x_normalizado2 = self.norm2(x_normalizado1 + saida_ffn)

        return x_normalizado2

class SOREModel_v2(nn.Module):
    def __init__(self, tamanho_vocab, dim_embed, tamanho_contexto, num_heads, num_layers):
        super(SOREModel_v2, self).__init__()

        self.token_embedding = nn.Embedding(num_embeddings=tamanho_vocab, embedding_dim=dim_embed)
        self.position_embedding = nn.Embedding(num_embeddings=tamanho_contexto, embedding_dim=dim_embed)
        self.blocks = nn.Sequential(
            *[Block(num_heads, dim_embed, dim_embed // num_heads) for _ in range(num_layers)]
        )

        self.lm_final = nn.LayerNorm(dim_embed)
        self.lm = nn.Linear(in_features=dim_embed, out_features=tamanho_vocab)

    def forward(self, x):

        B, T = x.shape # Batch size e Tamanho do contexto

        token_embeds = self.token_embedding(x)
        posicoes = torch.arange(T, device=x.device)
        pos_embeds = self.position_embedding(posicoes)

        embeds_com_posicao = token_embeds + pos_embeds

        x = self.blocks(embeds_com_posicao)

        x = self.lm_final(x)
        x = self.lm(x)

        return x
    

# --- Parâmetros para o SLM v2 (Transformer) ---
dim_embed = 32          # Dimensão do embedding (um pouco maior que 10)
num_heads = 4           # Número de cabeças de atenção
num_layers = 2          # Número de blocos Transformer empilhados
tamanho_contexto = 3    # Ainda estamos usando nosso contexto original
tamanho_vocabulario = len(vocabulario) # Do nosso tokenizador


modelo_v2 = SOREModel_v2(tamanho_vocabulario, dim_embed, tamanho_contexto, num_heads, num_layers)
funcao_de_perda_v2 = nn.CrossEntropyLoss()
otimizador_v2 = optim.AdamW(modelo_v2.parameters(), lr=0.001)

# --- Loop de Treinamento para o SLM v2 ---
print("\nIniciando treinamento do SLM v2 (Transformer)...")
for epoca in range(1000):
    logits = modelo_v2(X_tensor)
    logits_apenas_do_ultimo_token = logits[:, -1, :]
    perda = funcao_de_perda_v2(logits_apenas_do_ultimo_token, Y_tensor)

    otimizador_v2.zero_grad()
    perda.backward()
    otimizador_v2.step()

    if epoca % 100 == 0:
        print(f'Época {epoca}, Perda: {perda.item()}')

print(f"Treinamento do SLM v2 concluído. Perda final: {perda.item()}")

# --- Fase 2: Solve (ajuste fino) ---

print("\nFase 2: Ajuste fino do modelo com novos dados...")

for epoca in range(1000):
    logits = modelo_v2(X_tensor_final)
    logits_apenas_do_ultimo_token = logits[:, -1, :]
    perda = funcao_de_perda_v2(logits_apenas_do_ultimo_token, Y_tensor_final)

    otimizador_v2.zero_grad()
    perda.backward()
    otimizador_v2.step()

    if epoca % 100 == 0:
        print(f'Época {epoca}, Perda: {perda.item()}')

print(f"Ajuste fino concluído. Perda final: {perda.item()}")


def gerar_texto_v2(modelo, contexto_inicial, tamanho_de_contexto, quantos_novos_tokens):
    modelo.eval() # Coloca o modelo em modo de avaliação
    contexto_atual = contexto_inicial
    
    for _ in range(quantos_novos_tokens):
        # Preparar a entrada
        contexto_para_modelo = (contexto_atual[-tamanho_de_contexto:]).rjust(tamanho_de_contexto, ' ')
        tokens_entrada = codificar(contexto_para_modelo)
        tensor_entrada = torch.tensor([tokens_entrada], dtype=torch.long)
        
        # Fazer a previsão
        logits = modelo(tensor_entrada)
        logits_finais = logits[:, -1, :]
        probabilidades = torch.softmax(logits_finais, dim=1) 
        token_previsto_id = torch.argmax(probabilidades, dim=1).item()
        
        # Adicionar o token previsto ao contexto atual
        char_previsto = decodificar([token_previsto_id])
        contexto_atual += char_previsto

    return contexto_atual

# --- Teste do SLM v2 após ajuste fino ---
contexto_inicial = "olá"
contexto_gerado = gerar_texto_v2(modelo_v2, contexto_inicial, tamanho_contexto, 8)
print(f"Contexto gerado: {contexto_gerado}")

contexto_inicial = "oi"
contexto_gerado = gerar_texto_v2(modelo_v2, contexto_inicial, tamanho_contexto, 8)
print(f"Contexto gerado: {contexto_gerado}")
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
tamanho_de_contexto = 3
exemplos_x = []
exemplos_y = []

dados_codificados = codificar(texto)
for i in range(len(dados_codificados) - tamanho_de_contexto): 
    x = dados_codificados[i : i + tamanho_de_contexto]
    y = dados_codificados[i + tamanho_de_contexto]
    exemplos_x.append(x)
    exemplos_y.append(y)

X_tensor = torch.tensor(exemplos_x, dtype=torch.long)
Y_tensor = torch.tensor(exemplos_y, dtype=torch.long)

# --- Pilar 3, 4, 5: Modelo, Perda e Otimizador ---

class SOREModel(nn.Module):
    def __init__(self, tamanho_vocabulario, dimensao_embedding, tamanho_de_contexto):
        super(SOREModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=tamanho_vocabulario, embedding_dim=dimensao_embedding)
        dim_entrada_linear = tamanho_de_contexto * dimensao_embedding
        self.camada_linear = nn.Linear(in_features=dim_entrada_linear, out_features=tamanho_vocabulario)

    def forward(self, x):
        embeddings = self.embedding(x)
        embeddings_flat = embeddings.view(embeddings.shape[0], -1)
        logits = self.camada_linear(embeddings_flat)
        return logits

    def predict(self, x):
        logits = self.forward(x)
        # Aplicamos softmax nos logits para obter probabilidades
        probabilidades = torch.softmax(logits, dim=1) 
        # Escolhemos o token com a maior probabilidade (amostragem "gulosa")
        previsoes = torch.argmax(probabilidades, dim=1)
        return previsoes

# Instanciar tudo
dimensao_embedding = 10
modelo = SOREModel(tamanho_vocabulario, dimensao_embedding, tamanho_de_contexto)
funcao_de_perda = nn.CrossEntropyLoss()
otimizador = optim.AdamW(modelo.parameters(), lr=0.01) # Aumentei um pouco o lr para aprender mais rápido

# --- Loop de Treinamento ---
print("Iniciando treinamento...")
for epoca in range(1000):
    logits = modelo(X_tensor)
    perda = funcao_de_perda(logits, Y_tensor)

    otimizador.zero_grad()
    perda.backward()
    otimizador.step()

    if epoca % 100 == 0:
        print(f'Época {epoca}, Perda: {perda.item()}')
        
print(f"Treinamento concluído. Perda final: {perda.item()}")

# --- Geração de Texto ---
def gerar_texto(modelo, contexto_inicial, tamanho_de_contexto, quantos_novos_tokens):
    modelo.eval() # Coloca o modelo em modo de avaliação
    contexto_atual = contexto_inicial
    
    for _ in range(quantos_novos_tokens):
        # Preparar a entrada
        contexto_para_modelo = contexto_atual[-tamanho_de_contexto:]
        tokens_entrada = codificar(contexto_para_modelo)
        tensor_entrada = torch.tensor([tokens_entrada], dtype=torch.long)
        
        # Fazer a previsão
        tokens_previstos_tensor = modelo.predict(tensor_entrada)
        tokens_previstos_id = tokens_previstos_tensor.item()
        
        # Decodificar e atualizar
        char_previsto = decodificar([tokens_previstos_id])
        contexto_atual += char_previsto
        
    return contexto_atual

# --- Teste! ---
print("\n--- Gerando Texto ---")
contexto_inicial = "olá"
texto_gerado = gerar_texto(modelo, contexto_inicial, tamanho_de_contexto, 8)
print(f"Contexto: '{contexto_inicial}'")
print(f"Gerado:   '{texto_gerado}'")

dim_entrada = 10 # Dimensão do embedding
dim_saida_qkv = 10 # Dimensão de saída para Query, Key e Value

camada_query = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)
camada_key = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)
camada_value = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)

embbedings = [8,3,10] # Exemplo de tensor de embeddings com shape (batch_size=8, seq_len=3, dim_entrada=10)
queries = camada_query(embbedings)
keys = camada_key(embbedings)
values = camada_value(embbedings)

atencao = torch.matmul(queries, keys.transpose(-2, -1)) / (dim_saida_qkv ** 0.5)
pesos_atencao = F.softmax(atencao, dim=-1)

resultado_atencao = torch.matmul(pesos_atencao, values)


class atentionHead(nn.Module):
    def __init__(self, dim_entrada, dim_saida_qkv):
        super(atentionHead, self).__init__()
        self.camada_query = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)
        self.camada_key = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)
        self.camada_value = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)
        
    def forward(self, x):
        Q = self.camada_query(x)
        K = self.camada_key(x)
        V = self.camada_value(x)

        dim_k = K.size(-1)
        pontuacoes = torch.matmul(Q, K.transpose(-2, -1)) / (dim_k ** 0.5)
        pesos_atencao = F.softmax(pontuacoes, dim=-1)

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
        pass
import torch
import torch.nn as nn
import torch.nn.functional as F

class atentionHead(nn.Module):
    """Cabeça de Atenção implementada do zero"""
    def __init__(self, dim_entrada, dim_saida_qkv):
        super(atentionHead, self).__init__()                                                             # Inicializa a classe pai nn.Module
        self.camada_query = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)               # Camada linear para gerar Query
        self.camada_key = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)                 # Camada linear para gerar Key
        self.camada_value = nn.Linear(in_features=dim_entrada, out_features=dim_saida_qkv)               # Camada linear para gerar Value

    def forward(self, x):                                                                                # Define o método forward (Como os dados fluem pela rede)
        B, T, C = x.shape                                                                                 # Batch size, Tamanho da sequência, Dimensão de entrada

        Q = self.camada_query(x)                                                                         # Gera matriz Query
        K = self.camada_key(x)                                                                           # Gera matriz Key
        V = self.camada_value(x)                                                                         # Gera matriz Value

        dim_k = K.size(-1)                                                                               # Obtém a dimensão das chaves para escalonamento
        pontuacoes = torch.matmul(Q, K.transpose(-2, -1)) / (dim_k ** 0.5)                               # Calcula pontuações de atenção com escalonamento

        mascara = torch.tril(torch.ones(T, T)).to(x.device)                                              # Cria máscara triangular inferior para causalidade
        pontuacoes_mascaradas = pontuacoes.masked_fill(mascara == 0, float('-inf'))                     # Aplica máscara removendo pontuações futuras

        pesos_atencao = F.softmax(pontuacoes_mascaradas, dim=-1)                                         # Converte pontuações em pesos de atenção

        saida = torch.matmul(pesos_atencao, V)                                                           # Computa saída ponderada pelos valores

        return saida
    
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention implementada do zero"""
    def __init__(self, num_heads, dim_entrada, dim_saida_qkv):
        super(MultiHeadAttention, self).__init__()                                                          # Inicializa a classe pai nn.Module

        self.heads = nn.ModuleList([atentionHead(dim_entrada, dim_saida_qkv) for _ in range(num_heads)])    # Lista de cabeças de atenção

        self.projecao = nn.Linear(in_features=num_heads * dim_saida_qkv, out_features=dim_entrada)           # Camada de projeção linear final

    def forward(self, x):                                                                                    # Define o método forward (Como os dados fluem pela rede)
        saidas_cabecas = [head(x) for head in self.heads]                                                    # Processa cada cabeça de atenção
        saida_concatenada = torch.cat(saidas_cabecas, dim=-1)                                                # Concatena as saídas das cabeças

        saida_final = self.projecao(saida_concatenada)                                                       # Aplica projeção linear final

        return saida_final
    
class Block(nn.Module):
    """Bloco Transformer com Self-Attention e Feed-Forward"""
    def __init__(self, num_heads, dim_entrada, dim_saida_qkv):
        super(Block, self).__init__()                                                                       # Inicializa a classe pai nn.Module

        # --- Self-Attention ---
        self.atencao = MultiHeadAttention(num_heads, dim_entrada, dim_saida_qkv)                             # Camada Multi-Head Attention
        self.norm1 = nn.LayerNorm(dim_entrada)                                                               # Normalização após atenção

        # --- Feed-Forward ---
        dim_ffn = 4 * dim_entrada                                                                            # Dimensão da rede feed-forward (4x a entrada)

        self.ffn = nn.Sequential(                                                                            # Rede feed-forward de duas camadas
            nn.Linear(in_features=dim_entrada, out_features=dim_ffn),                                        # Primeira camada linear (expansão)
            nn.ReLU(),                                                                                       # Função de ativação ReLU
            nn.Linear(in_features=dim_ffn, out_features=dim_entrada)                                         # Segunda camada linear (projeção)
        )
        self.norm2 = nn.LayerNorm(dim_entrada)                                                               # Normalização após feed-forward

    def forward(self, x):                                                                                    # Define o método forward (Como os dados fluem pela rede)
        saida_atencao = self.atencao(x)                                                                      # Processa através da Multi-Head Attention
        x_normalizado1 = self.norm1(x + saida_atencao)                                                       # Adiciona residual e normaliza

        saida_ffn = self.ffn(x_normalizado1)                                                                 # Processa através da rede feed-forward
        x_normalizado2 = self.norm2(x_normalizado1 + saida_ffn)                                              # Adiciona residual e normaliza

        return x_normalizado2

# --- Classe do Modelo v2 (Transformer) ---
class SOREModel_v2(nn.Module):
    """Modelo Transformer para geração de texto - Versão 2 do SOREModel"""
    def __init__(self, tamanho_vocab, dim_embed, tamanho_contexto, num_heads, num_layers):
        super(SOREModel_v2, self).__init__()                                                                 # Inicializa a classe pai nn.Module

        self.tamanho_contexto = tamanho_contexto                                                             # Armazena o tamanho do contexto
        self.token_embedding = nn.Embedding(num_embeddings=tamanho_vocab, embedding_dim=dim_embed)           # Embedding para tokens do vocabulário
        self.position_embedding = nn.Embedding(num_embeddings=tamanho_contexto, embedding_dim=dim_embed)     # Embedding para posições
        self.blocks = nn.Sequential(                                                                         # Sequência de blocos Transformer
            *[Block(num_heads, dim_embed, dim_embed // num_heads) for _ in range(num_layers)]
        )

        self.lm_final = nn.LayerNorm(dim_embed)                                                              # Normalização final antes da saída
        self.lm = nn.Linear(in_features=dim_embed, out_features=tamanho_vocab)                               # Camada linear final para prever próximos tokens

    def forward(self, x):                                                                                    # Define o método forward (Como os dados fluem pela rede)
        B, T = x.shape                                                                                       # Batch size e Tamanho do contexto

        token_embeds = self.token_embedding(x)                                                               # Obtém embeddings dos tokens
        posicoes = torch.arange(T, device=x.device)                                                          # Gera índices das posições
        pos_embeds = self.position_embedding(posicoes)                                                       # Obtém embeddings das posições

        embeds_com_posicao = token_embeds + pos_embeds                                                       # Combina embeddings de tokens e posições

        x = self.blocks(embeds_com_posicao)                                                                  # Processa através dos blocos Transformer

        x = self.lm_final(x)                                                                                 # Aplica normalização final
        x = self.lm(x)                                                                                       # Gera logits para cada token do vocabulário

        return x
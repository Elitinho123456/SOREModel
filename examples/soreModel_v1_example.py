import torch
import torch.nn as nn

# --- Classe do Modelo ---
class SOREModel(nn.Module):

    def __init__(self, tamanho_vocabulario, dimensao_embedding, tamanho_de_contexto):
        super(SOREModel, self).__init__()                                                                       # Inicializa a classe pai nn.Module
    
        self.embedding = nn.Embedding(num_embeddings=tamanho_vocabulario, embedding_dim=dimensao_embedding)     # Tabela de embedding
        dim_entrada_linear = tamanho_de_contexto * dimensao_embedding                                           # Calcula a dimensão de entrada para a camada linear
        self.camada_linear = nn.Linear(in_features=dim_entrada_linear, out_features=tamanho_vocabulario)        # Define a camada linear
    
    def forward(self, x):                                                                                       # Define o método forward(Como os dados Fluem pela rede)
        # X é um tensor de índices de forma (batch_size, tamanho_de_contexto)   
        embeddings = self.embedding(x)                                                                          # Obtém os embeddings para os índices de entrada
        embeddings_flat = embeddings.view(embeddings.shape[0], -1)                                              # Achata os embeddings para a camada linear
        logits = self.camada_linear(embeddings_flat)                                                            # Passa os embeddings achatados pela camada linear
    
        return logits   
    
    def predict(self, x):   
        logits = self.forward(x)                                                                                # Obtém os logits do método forward
        probabilidades = torch.softmax(logits, dim=1)                                                           # Aplica softmax para obter probabilidades
        previsoes = torch.argmax(probabilidades, dim=1)                                                         # Obtém as previsões como o índice da maior probabilidade
    
        return previsoes
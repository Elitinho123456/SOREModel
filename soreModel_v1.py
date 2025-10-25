import torch
import torch.nn as nn

class SOREModel(nn.Module):

    def __init__(self, tamanho_vocabulario, dimensao_embedding, tamanho_de_contexto):
        super(SOREModel, self).__init__()                                                                   # Inicializa a classe pai nn.Module

        self.embedding = nn.Embedding(num_embeddings=tamanho_vocabulario, embedding_dim=dimensao_embedding) # Tabela de embedding
        dim_entrada_linear = tamanho_de_contexto * dimensao_embedding                                       # Calcula a dimensão de entrada para a camada linear
        self.camada_linear = nn.Linear(in_features=dim_entrada_linear, out_features=tamanho_vocabulario)    # Define a camada linear

    def forward(self, x):                                                                                   # Define o método forward(Como os dados Fluem pela rede)
        # X é um tensor de índices de forma (batch_size, tamanho_de_contexto)
        embeddings = self.embedding(x)                                                                      # Obtém os embeddings para os índices de entrada
        embeddings_flat = embeddings.view(embeddings.shape[0], -1)                                          # Achata os embeddings para a camada linear
        logits = self.camada_linear(embeddings_flat)                                                        # Passa os embeddings achatados pela camada linear

        return logits

    def predict(self, x):
        logits = self.forward(x)                                                                            # Obtém os logits do método forward
        probabilidades = torch.softmax(logits, dim=1)                                                       # Aplica softmax para obter probabilidades
        previsoes = torch.argmax(probabilidades, dim=1)                                                     # Obtém as previsões como o índice da maior probabilidade

        return previsoes
    

# class SOREModel(nn.Module):
#     def __init__(self, tamanho_vocabulario, dimensao_embedding, tamanho_de_contexto):
#         super(SOREModel, self).__init__()
#         self.embedding = nn.Embedding(num_embeddings=tamanho_vocabulario, embedding_dim=dimensao_embedding)
#         dim_entrada_linear = tamanho_de_contexto * dimensao_embedding
#         self.camada_linear = nn.Linear(in_features=dim_entrada_linear, out_features=tamanho_vocabulario)

#     def forward(self, x):
#         embeddings = self.embedding(x)
#         embeddings_flat = embeddings.view(embeddings.shape[0], -1)
#         logits = self.camada_linear(embeddings_flat)
#         return logits

#     def predict(self, x):
#         logits = self.forward(x)
#         # Aplicamos softmax nos logits para obter probabilidades
#         probabilidades = torch.softmax(logits, dim=1) 
#         # Escolhemos o token com a maior probabilidade (amostragem "gulosa")
#         previsoes = torch.argmax(probabilidades, dim=1)
#         return previsoes

# # Instanciar tudo
# dimensao_embedding = 10
# modelo = SOREModel(tamanho_vocabulario, dimensao_embedding, tamanho_de_contexto)
# funcao_de_perda = nn.CrossEntropyLoss()
# otimizador = optim.AdamW(modelo.parameters(), lr=0.01) # Aumentei um pouco o lr para aprender mais rápido

# # --- Loop de Treinamento ---
# print("Iniciando treinamento...")
# for epoca in range(1000):
#     logits = modelo(X_tensor)
#     perda = funcao_de_perda(logits, Y_tensor)

#     otimizador.zero_grad()
#     perda.backward()
#     otimizador.step()

#     if epoca % 100 == 0:
#         print(f'Época {epoca}, Perda: {perda.item()}')
        
# print(f"Treinamento concluído. Perda final: {perda.item()}")

# # --- Geração de Texto ---
# def gerar_texto(modelo, contexto_inicial, tamanho_de_contexto, quantos_novos_tokens):
#     modelo.eval() # Coloca o modelo em modo de avaliação
#     contexto_atual = contexto_inicial
    
#     for _ in range(quantos_novos_tokens):
#         # Preparar a entrada
#         contexto_para_modelo = contexto_atual[-tamanho_de_contexto:]
#         tokens_entrada = codificar(contexto_para_modelo)
#         tensor_entrada = torch.tensor([tokens_entrada], dtype=torch.long)
        
#         # Fazer a previsão
#         tokens_previstos_tensor = modelo.predict(tensor_entrada)
#         tokens_previstos_id = tokens_previstos_tensor.item()
        
#         # Decodificar e atualizar
#         char_previsto = decodificar([tokens_previstos_id])
#         contexto_atual += char_previsto
        
#     return contexto_atual

# # --- Teste! ---
# print("\n--- Gerando Texto ---")
# contexto_inicial = "olá"
# texto_gerado = gerar_texto(modelo, contexto_inicial, tamanho_de_contexto, 8)
# print(f"Contexto: '{contexto_inicial}'")
# print(f"Gerado:   '{texto_gerado}'")
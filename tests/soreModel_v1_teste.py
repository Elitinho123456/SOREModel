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
"""
Módulo de Treinamento para o SOREModel
Implementa funções para treinamento e ajuste fino do modelo
"""

import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    """Classe para gerenciar o treinamento do modelo"""

    def __init__(self, modelo, tokenizer, device=None):
        """
        Inicializa o trainer

        Args:
            modelo: Modelo SOREModel a ser treinado
            tokenizer: Tokenizer para processar os dados
            device: Dispositivo para treinamento (cpu ou cuda)
        """
        self.modelo = modelo
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo.to(self.device)

        # Configurações padrão
        self.funcao_perda = nn.CrossEntropyLoss()
        self.otimizador = None
        self.historico_perdas = []

    def configurar_otimizador(self, learning_rate=0.001, weight_decay=0.01):
        """
        Configura o otimizador AdamW

        Args:
            learning_rate (float): Taxa de aprendizado
            weight_decay (float): Decaimento de peso para regularização
        """
        self.otimizador = optim.AdamW(
            self.modelo.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def preparar_dados(self, textos, contexto_tamanho):
        """
        Prepara os dados de treinamento

        Args:
            textos (list): Lista de textos para treinamento
            contexto_tamanho (int): Tamanho do contexto

        Returns:
            tuple: (X_tensor, Y_tensor) - Tensores preparados
        """
        exemplos_x, exemplos_y = self.tokenizer.encode_batch(textos, contexto_tamanho)

        X_tensor = torch.tensor(exemplos_x, dtype=torch.long)
        Y_tensor = torch.tensor(exemplos_y, dtype=torch.long)

        return X_tensor.to(self.device), Y_tensor.to(self.device)

    def treinar(self, textos, contexto_tamanho, num_epocas=1000, batch_size=32, learning_rate=0.001):
        """
        Treina o modelo com os dados fornecidos

        Args:
            textos (list): Lista de textos para treinamento
            contexto_tamanho (int): Tamanho do contexto
            num_epocas (int): Número de épocas de treinamento
            batch_size (int): Tamanho do batch
            learning_rate (float): Taxa de aprendizado
        """
        # Preparar dados
        X_tensor, Y_tensor = self.preparar_dados(textos, contexto_tamanho)

        # Configurar otimizador
        self.configurar_otimizador(learning_rate)

        print(f"Iniciando treinamento com {len(X_tensor)} exemplos...")
        print(f"Dispositivo: {self.device}")

        self.modelo.train()

        for epoca in range(num_epocas):
            # Embaralhar dados
            indices = torch.randperm(len(X_tensor))
            X_embaralhado = X_tensor[indices]
            Y_embaralhado = Y_tensor[indices]

            perda_epoca = 0
            num_batches = 0

            # Treinamento em mini-batches
            for i in range(0, len(X_embaralhado), batch_size):
                batch_X = X_embaralhado[i:i+batch_size]
                batch_Y = Y_embaralhado[i:i+batch_size]

                # Forward pass
                logits = self.modelo(batch_X)
                logits_ultimo_token = logits[:, -1, :]  # Apenas o último token

                perda = self.funcao_perda(logits_ultimo_token, batch_Y)

                # Backward pass
                self.otimizador.zero_grad()
                perda.backward()
                self.otimizador.step()

                perda_epoca += perda.item()
                num_batches += 1

            # Calcular perda média da época
            perda_media = perda_epoca / num_batches
            self.historico_perdas.append(perda_media)

            # Log progress
            if (epoca + 1) % 100 == 0:
                print(f'Época {epoca+1}/{num_epocas}, Perda: {perda_media:.4f}')

        print(f"Treinamento concluído. Perda final: {self.historico_perdas[-1]:.4f}")

    def ajustar_fino(self, textos_novos, contexto_tamanho, num_epocas=1000, batch_size=32):
        """
        Ajuste fino do modelo com novos dados

        Args:
            textos_novos (list): Novos textos para ajuste fino
            contexto_tamanho (int): Tamanho do contexto
            num_epocas (int): Número de épocas de ajuste fino
            batch_size (int): Tamanho do batch
        """
        print("\nIniciando ajuste fino...")

        # Preparar novos dados
        X_novo, Y_novo = self.preparar_dados(textos_novos, contexto_tamanho)

        # Combinar com dados anteriores se existirem
        if hasattr(self, 'X_anterior') and hasattr(self, 'Y_anterior'):
            X_combinado = torch.cat([self.X_anterior, X_novo], dim=0)
            Y_combinado = torch.cat([self.Y_anterior, Y_novo], dim=0)
        else:
            X_combinado, Y_combinado = X_novo, Y_novo

        # Salvar dados anteriores para próximas iterações
        self.X_anterior = X_combinado
        self.Y_anterior = Y_combinado

        print(f"Ajuste fino com {len(X_combinado)} exemplos totais...")

        self.modelo.train()

        for epoca in range(num_epocas):
            # Embaralhar dados
            indices = torch.randperm(len(X_combinado))
            X_embaralhado = X_combinado[indices]
            Y_embaralhado = Y_combinado[indices]

            perda_epoca = 0
            num_batches = 0

            # Ajuste fino em mini-batches
            for i in range(0, len(X_embaralhado), batch_size):
                batch_X = X_embaralhado[i:i+batch_size]
                batch_Y = Y_embaralhado[i:i+batch_size]

                # Forward pass
                logits = self.modelo(batch_X)
                logits_ultimo_token = logits[:, -1, :]

                perda = self.funcao_perda(logits_ultimo_token, batch_Y)

                # Backward pass
                self.otimizador.zero_grad()
                perda.backward()
                self.otimizador.step()

                perda_epoca += perda.item()
                num_batches += 1

            # Calcular perda média da época
            perda_media = perda_epoca / num_batches
            self.historico_perdas.append(perda_media)

            # Log progress
            if (epoca + 1) % 100 == 0:
                print(f'Época {epoca+1}/{num_epocas}, Perda: {perda_media:.4f}')

        print(f"Ajuste fino concluído. Perda final: {self.historico_perdas[-1]:.4f}")

    def salvar_modelo(self, caminho):
        """Salva o modelo treinado"""
        torch.save({
            'model_state_dict': self.modelo.state_dict(),
            'optimizer_state_dict': self.otimizador.state_dict() if self.otimizador else None,
            'historico_perdas': self.historico_perdas,
        }, caminho)
        print(f"Modelo salvo em: {caminho}")

    def carregar_modelo(self, caminho):
        """Carrega um modelo salvo"""
        checkpoint = torch.load(caminho, map_location=self.device)
        self.modelo.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] and self.otimizador:
            self.otimizador.load_state_dict(checkpoint['optimizer_state_dict'])
        self.historico_perdas = checkpoint.get('historico_perdas', [])
        print(f"Modelo carregado de: {caminho}")

    def get_historico_perdas(self):
        """Retorna o histórico de perdas"""
        return self.historico_perdas.copy()

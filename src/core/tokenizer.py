"""
Módulo de Tokenização para o SOREModel
Implementa funcionalidades de codificação e decodificação de texto
"""

import numpy as np
from utils import criar_vocabulario_de_caracteres


class Tokenizer:
    """Tokenizer simples para caracteres"""

    def __init__(self, vocabulario=None):
        """
        Inicializa o tokenizer

        Args:
            vocabulario (list): Lista de caracteres únicos para formar o vocabulário.
                              Se None, usa todos os caracteres imprimíveis.
        """
        if vocabulario is None:
            # Define um vocabulário padrão com caracteres comuns
            self.modelo_conhecimento = criar_vocabulario_de_caracteres()

        else:
            self.modelo_conhecimento = vocabulario

        # Remove duplicatas e ordena
        self.vocabulario = sorted(list(set(self.modelo_conhecimento)))

        # Dicionários de mapeamento
        self.stoi = {char: i for i, char in enumerate(self.vocabulario)}                                    # String to Index
        self.itos = {i: char for i, char in enumerate(self.vocabulario)}                                    # Index to String

        print(f"Tokenizer inicializado com vocabulário de {len(self.vocabulario)} caracteres únicos")

    def codificar(self, texto):
        """
        Codifica um texto em uma lista de índices

        Args:
            texto (str): Texto a ser codificado

        Returns:
            list: Lista de índices correspondentes aos caracteres
        """
        try:
            tokens = []
            for char in texto:
                if char in self.stoi:
                    tokens.append(self.stoi[char])
                else:
                    print(f"Aviso: caracter '{repr(char)}' ignorado (não está no vocabulário)")
                    continue
            return tokens
        except KeyError as e:
            raise ValueError(f"Caracter '{e.args[0]}' não encontrado no vocabulário")

    def decodificar(self, indices):
        """
        Decodifica uma lista de índices em texto

        Args:
            indices (list): Lista de índices a serem decodificados

        Returns:
            str: Texto decodificado
        """
        try:
            return ''.join([self.itos[i] for i in indices])
        except KeyError as e:
            raise ValueError(f"Índice {e.args[0]} não encontrado no vocabulário")

    def get_vocab_size(self):
        """Retorna o tamanho do vocabulário"""
        return len(self.vocabulario)

    def get_vocab(self):
        """Retorna o vocabulário como lista"""
        return self.vocabulario.copy()

    def encode_batch(self, textos, contexto_tamanho):
        """
        Codifica múltiplos textos e gera exemplos de treinamento (Versão otimizada com NumPy)

        Args:
            textos (list): Lista de textos para processar
            contexto_tamanho (int): Tamanho do contexto para cada exemplo

        Returns:
            tuple: (np_x, np_y) - Arrays NumPy de entrada e saída
        """
        tokens_juntos = []
        for texto in textos:
            tokens_juntos.extend(self.codificar(texto))
        
        dados_np = np.array(tokens_juntos, dtype=np.int64)

        num_exemplos = len(dados_np) - contexto_tamanho
        if num_exemplos <= 0:
            print("Aviso: Dados insuficientes para criar exemplos com o contexto fornecido.")
            return np.array([]), np.array([]) # Retorna arrays vazios

        np_x = np.zeros((num_exemplos, contexto_tamanho), dtype=np.int64)
        np_y = np.zeros(num_exemplos, dtype=np.int64)

        for i in range(num_exemplos):
            np_x[i] = dados_np[i : i + contexto_tamanho]
            np_y[i] = dados_np[i + contexto_tamanho]

        return np_x, np_y


def criar_tokenizer_basico(simbolos=None):
    """Cria um tokenizer com vocabulário básico otimizado"""

    if not simbolos:
        print("Nenhum símbolo adicional fornecido. Usando vocabulário básico padrão.")

        caracteres_comuns = criar_vocabulario_de_caracteres()
    else:
        caracteres_comuns = simbolos

    return Tokenizer(caracteres_comuns)
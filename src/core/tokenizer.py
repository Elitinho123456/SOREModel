"""
Módulo de Tokenização para o SOREModel
Implementa funcionalidades de codificação e decodificação de texto
"""

from tokenizers import Tokenizer as TTK
import numpy as np

TTKconfig = 'tokenizer/sore_bpe_tokenizer.json'

class Tokenizer:
    def __init__(self, tokenizer_path=TTKconfig):
        self.tokenizer = TTK.from_file(tokenizer_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self):
        """Retorna o novo tamanho do vocabulário BPE."""
        return self.tokenizer.get_vocab_size()

    def encode_batch(self, textos, contexto_tamanho):
        """
        Codifica uma lista de textos (batch) e gera os pares (X, Y)
        para o treinamento do Transformer.
        """
        # 1. Tokeniza todos os textos de uma vez (super rápido)
        # O .encode_batch já lida com o 'padding' para nós
        encodings = self.tokenizer.encode_batch(textos)

        tokens_juntos = []
        for enc in encodings:
            # Pegamos os IDs (números) de cada texto
            tokens_juntos.extend(enc.ids)
        
        # 2. O resto do código é idêntico ao seu 'tokenizer.py' antigo
        #    Usamos numpy para criar os exemplos (X, Y)
        
        dados_np = np.array(tokens_juntos, dtype=np.int64)

        num_exemplos = len(dados_np) - contexto_tamanho
        if num_exemplos <= 0:
            print("Aviso: Dados insuficientes para criar exemplos.")
            return np.array([]), np.array([])

        np_x = np.zeros((num_exemplos, contexto_tamanho), dtype=np.int64)
        np_y = np.zeros(num_exemplos, dtype=np.int64)

        for i in range(num_exemplos):
            np_x[i] = dados_np[i : i + contexto_tamanho]
            np_y[i] = dados_np[i + contexto_tamanho]

        return np_x, np_y
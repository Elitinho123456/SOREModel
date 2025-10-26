"""
Módulo de Tokenização para o SOREModel
Implementa funcionalidades de codificação e decodificação de texto
"""

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
            self.modelo_conhecimento = [
                'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                'Á','À','Ã','Â','Ä','É','È','Ê','Ë','Í','Ì','Î','Ï','Ó','Ò','Õ','Ô','Ö','Ú','Ù','Û','Ü','Ç',
                'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
                'á','à','ã','â','ä','é','è','ê','ë','í','ì','î','ï','ó','ò','õ','ô','ö','ú','ù','û','ü','ç',
                ' ',' ',' ',' ',' ',' ',  # Espaços múltiplos para dar mais peso
                ',','!','.','?',';','\n'
            ]
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
            return [self.stoi[char] for char in texto]
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
            raise ValueError(f"Índice '{e.args[0]}' não encontrado no vocabulário")

    def get_vocab_size(self):
        """Retorna o tamanho do vocabulário"""
        return len(self.vocabulario)

    def get_vocab(self):
        """Retorna o vocabulário como lista"""
        return self.vocabulario.copy()

    def encode_batch(self, textos, contexto_tamanho):
        """
        Codifica múltiplos textos e gera exemplos de treinamento

        Args:
            textos (list): Lista de textos para processar
            contexto_tamanho (int): Tamanho do contexto para cada exemplo

        Returns:
            tuple: (X_tensor, Y_tensor) - Tensores de entrada e saída
        """
        exemplos_x = []
        exemplos_y = []

        for texto in textos:
            dados_codificados = self.codificar(texto)
            for i in range(len(dados_codificados) - contexto_tamanho):
                x = dados_codificados[i : i + contexto_tamanho]
                y = dados_codificados[i + contexto_tamanho]
                exemplos_x.append(x)
                exemplos_y.append(y)

        return exemplos_x, exemplos_y


def criar_tokenizer_basico():
    """Cria um tokenizer com vocabulário básico otimizado"""
    caracteres_comuns = [
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
        'á','à','ã','â','ä','é','è','ê','ë','í','ì','î','ï','ó','ò','õ','ô','ö','ú','ù','û','ü','ç',
        ' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',  # Múltiplos espaços
        ',','!','.','?',';','\n','\t'
    ]
    return Tokenizer(caracteres_comuns)

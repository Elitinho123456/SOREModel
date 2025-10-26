"""
Módulo de Geração para o SOREModel
Implementa funcionalidades para geração de texto usando o modelo treinado
"""

import torch

class TextGenerator:
    """Classe para geração de texto usando o modelo SOREModel"""

    def __init__(self, modelo, tokenizer, device=None):
        """
        Inicializa o gerador de texto

        Args:
            modelo: Modelo SOREModel treinado
            tokenizer: Tokenizer para processar o texto
            device: Dispositivo para inferência (cpu ou cuda)
        """
        self.modelo = modelo
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.modelo.to(self.device)
        self.modelo.eval()  # Coloca em modo de avaliação

    def gerar_texto(self, contexto_inicial, max_length=50, temperature=1.0, top_k=None, top_p=None):
        """
        Gera texto baseado no contexto inicial

        Args:
            contexto_inicial (str): Texto inicial para começar a geração
            max_length (int): Número máximo de tokens a gerar
            temperature (float): Temperatura para amostragem (1.0 = sem alteração)
            top_k (int): Se especificado, apenas os top_k tokens mais prováveis são considerados
            top_p (float): Se especificado, apenas tokens que somam probabilidade top_p são considerados

        Returns:
            str: Texto gerado completo
        """
        contexto_atual = contexto_inicial

        with torch.no_grad():
            for _ in range(max_length):
                # Preparar entrada
                contexto_codificado = self.tokenizer.codificar(contexto_atual)
                if len(contexto_codificado) == 0:
                    break

                # Usar apenas os últimos tokens se o contexto for muito longo
                max_context = getattr(self.modelo, 'tamanho_contexto', len(contexto_codificado))
                if len(contexto_codificado) > max_context:
                    contexto_codificado = contexto_codificado[-max_context:]

                # Converter para tensor
                tensor_entrada = torch.tensor([contexto_codificado], dtype=torch.long).to(self.device)

                # Fazer previsão
                logits = self.modelo(tensor_entrada)
                logits_ultimo_token = logits[:, -1, :]

                # Aplicar temperatura
                if temperature != 1.0:
                    logits_ultimo_token = logits_ultimo_token / temperature

                # Aplicar top-k filtering
                if top_k is not None:
                    indices_to_remove = logits_ultimo_token < torch.topk(logits_ultimo_token, top_k)[0][..., -1, None]
                    logits_ultimo_token[indices_to_remove] = float('-inf')

                # Aplicar top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits_ultimo_token, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens com probabilidade cumulativa acima de top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                    logits_ultimo_token[indices_to_remove] = float('-inf')

                # Converter para probabilidades
                probs = torch.softmax(logits_ultimo_token, dim=-1)

                # Amostrar próximo token
                next_token_idx = torch.multinomial(probs, num_samples=1).item()

                # Decodificar e adicionar ao contexto
                next_char = self.tokenizer.decodificar([next_token_idx])
                contexto_atual += next_char

                # Parar se gerar token de fim (se existir)
                if next_char in ['\n', '\0']:
                    break

        return contexto_atual

    def gerar_texto_beam_search(self, contexto_inicial, max_length=50, beam_width=3):
        """
        Gera texto usando beam search para melhor qualidade

        Args:
            contexto_inicial (str): Texto inicial para começar a geração
            max_length (int): Número máximo de tokens a gerar
            beam_width (int): Largura do beam (número de sequências a manter)

        Returns:
            str: Melhor texto gerado
        """
        # Implementação simplificada - em produção seria mais otimizada
        candidatos = [contexto_inicial]

        with torch.no_grad():
            for _ in range(max_length):
                novos_candidatos = []

                for candidato in candidatos:
                    if len(candidato) == 0:
                        continue

                    # Preparar entrada
                    contexto_codificado = self.tokenizer.codificar(candidato)
                    max_context = getattr(self.modelo, 'tamanho_contexto', len(contexto_codificado))
                    if len(contexto_codificado) > max_context:
                        contexto_codificado = contexto_codificado[-max_context:]

                    tensor_entrada = torch.tensor([contexto_codificado], dtype=torch.long).to(self.device)

                    # Fazer previsão
                    logits = self.modelo(tensor_entrada)
                    logits_ultimo_token = logits[:, -1, :]

                    # Pegar top beam_width tokens
                    probs = torch.softmax(logits_ultimo_token, dim=-1)
                    top_probs, top_indices = torch.topk(probs, beam_width)

                    for i in range(beam_width):
                        prob = top_probs[0, i].item()
                        token_idx = top_indices[0, i].item()
                        next_char = self.tokenizer.decodificar([token_idx])

                        if prob > 0:  # Só adicionar se probabilidade for significativa
                            novos_candidatos.append((candidato + next_char, prob))

                # Manter apenas os beam_width melhores candidatos
                if len(novos_candidatos) > beam_width:
                    novos_candidatos.sort(key=lambda x: x[1], reverse=True)
                    candidatos = [c[0] for c in novos_candidatos[:beam_width]]
                else:
                    candidatos = [c[0] for c in novos_candidatos]

                if not candidatos:
                    break

        # Retornar o melhor candidato (primeiro da lista ordenada por probabilidade)
        return candidatos[0] if candidatos else contexto_inicial

    def completar_texto(self, texto_incompleto, max_completar=20):
        """
        Completa um texto dado

        Args:
            texto_incompleto (str): Texto a ser completado
            max_completar (int): Número máximo de tokens para completar

        Returns:
            str: Texto completo
        """
        return self.gerar_texto(texto_incompleto, max_length=max_completar)

    def gerar_variacoes(self, contexto_inicial, num_variacoes=5, max_length=30, temperature=1.2):
        """
        Gera múltiplas variações de um contexto

        Args:
            contexto_inicial (str): Texto base para variações
            num_variacoes (int): Número de variações a gerar
            max_length (int): Comprimento máximo de cada variação
            temperature (float): Temperatura para mais diversidade

        Returns:
            list: Lista de textos gerados
        """
        variacoes = []
        for _ in range(num_variacoes):
            variacao = self.gerar_texto(
                contexto_inicial,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9  # Para manter coerência
            )
            variacoes.append(variacao)

        return variacoes

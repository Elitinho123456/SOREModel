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
                # Codificar o texto usando o tokenizer
                try:
                    # Tenta usar o método encode do tokenizer (para tokenizers do HuggingFace)
                    if hasattr(self.tokenizer, 'encode'):
                        encoding = self.tokenizer.encode(contexto_atual)
                        contexto_codificado = encoding.ids
                    # Se não tiver encode, tenta chamar diretamente (para compatibilidade com versões mais antigas)
                    else:
                        encoding = self.tokenizer(contexto_atual)
                        contexto_codificado = encoding.input_ids
                except Exception as e:
                    print(f"Erro ao codificar o texto: {e}")
                    break

                if len(contexto_codificado) == 0:
                    break

                # Usar apenas os últimos tokens se o contexto for muito longo
                max_context = getattr(self.modelo, 'cfg', None)
                if max_context is not None and hasattr(max_context, 'context_size'):
                    max_context = max_context.context_size
                else:
                    max_context = 1024  # Valor padrão
                
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
                try:
                    # Usa o método de decodificação do tokenizador, que é mais robusto
                    token_gerado = self.tokenizer.decode([next_token_idx], skip_special_tokens=False)

                    # O tokenizador ByteLevel BPE pode gerar o prefixo 'Ġ' para espaços.
                    # Esta lógica ajuda a formatar o texto de forma mais limpa.
                    if token_gerado.startswith('Ġ'):
                        # Adiciona um espaço e o resto do token
                        contexto_atual += ' ' + token_gerado[1:]
                    else:
                        contexto_atual += token_gerado

                except Exception as e:
                    print(f"\nERRO ao decodificar o token {next_token_idx}: {e}")
                    break

                # Parar se gerar token de fim (se existir)
                if token_gerado in ['\n', '\0']:
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
        # Versão otimizada: processamento em batch dos beams não finalizados,
        # uso de log-probs e manutenção de beams finalizados como candidatos.

        # Codificar contexto inicial
        try:
            if hasattr(self.tokenizer, 'encode'):
                encoding = self.tokenizer.encode(contexto_inicial)
                encoded_context = encoding.ids
            else:
                encoding = self.tokenizer(contexto_atual)
                encoded_context = encoding.input_ids
        except Exception as e:
            print(f"Erro ao codificar o texto: {e}")
            return contexto_inicial

        if len(encoded_context) == 0:
            return contexto_inicial

        # Configurações úteis
        device = self.device
        model = self.modelo
        
        # Obter tamanho máximo do contexto da configuração do modelo
        model_config = getattr(model, 'cfg', None)
        if model_config is not None and hasattr(model_config, 'context_size'):
            max_context = model_config.context_size
        else:
            max_context = 1024  # Valor padrão
            
        # Obter IDs especiais do tokenizer
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        eos_id = getattr(self.tokenizer, 'eos_token_id', None)

        # Inicializar beams como lista de (token_list, score, finished_flag)
        beams = [(encoded_context[:max_context] if max_context else encoded_context, 0.0, False)]

        with torch.no_grad():
            for _ in range(max_length):
                # Separar beams finalizados e não finalizados
                finished_candidates = []
                to_expand_idx = []
                for i, (seq, score, finished) in enumerate(beams):
                    if finished:
                        # manter como candidato estático
                        finished_candidates.append((seq, score, True))
                    else:
                        to_expand_idx.append(i)

                # Se não houver beams para expandir, terminar
                if not to_expand_idx:
                    # escolher o melhor entre os finalizados
                    best = max(finished_candidates, key=lambda x: x[1]) if finished_candidates else beams[0]
                    return self.tokenizer.decodificar(best[0])

                # Preparar batch das sequências a expandir
                batch_seqs = [beams[i][0] for i in to_expand_idx]
                lengths = [len(s) for s in batch_seqs]
                batch_max_len = max(lengths)

                # Truncar se necessário ao tamanho do contexto do modelo
                if max_context and batch_max_len > max_context:
                    # manter apenas últimos tokens para cada sequência
                    batch_seqs = [s[-max_context:] for s in batch_seqs]
                    lengths = [len(s) for s in batch_seqs]
                    batch_max_len = max(lengths)

                # Criar tensor padded (direção: right padding)
                batch_tensor = torch.full((len(batch_seqs), batch_max_len), pad_id, dtype=torch.long, device=device)
                for i, s in enumerate(batch_seqs):
                    if len(s) > 0:
                        batch_tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)

                # Chamar o modelo em batch
                logits = model(batch_tensor)  # (batch, seq_len, vocab)
                # obter logits do último token real para cada entrada
                idxs = torch.tensor([l - 1 for l in lengths], device=device)
                batch_idx = torch.arange(len(batch_seqs), device=device)
                last_logits = logits[batch_idx, idxs, :]  # (batch, vocab)

                # log-probabilidades
                log_probs = torch.log_softmax(last_logits, dim=-1)  # (batch, vocab)

                # Pegar topk por sequência (beam_width)
                topk_vals, topk_inds = torch.topk(log_probs, k=beam_width, dim=-1)  # ambos (batch, beam_width)

                # Construir candidatos: combinar beams finalizados + expansões
                candidates = []
                # adicionar beams finalizados como candidatos diretos
                for seq, score, finished in finished_candidates:
                    candidates.append((seq, score, True))

                # adicionar expansões das sequências não finalizadas
                for local_i, global_i in enumerate(to_expand_idx):
                    orig_seq, orig_score, _ = beams[global_i]
                    for k in range(topk_inds.size(1)):
                        token_id = int(topk_inds[local_i, k].item())
                        token_logprob = float(topk_vals[local_i, k].item())
                        new_seq = orig_seq + [token_id]
                        new_score = orig_score + token_logprob
                        # verificar se este token finalizou a sequência
                        is_finished = False
                        if eos_id is not None:
                            is_finished = (token_id == eos_id)
                        else:
                            # fallback: decodificar token isolado e checar caracteres de parada
                            try:
                                if hasattr(self.tokenizer, 'decode'):
                                    token_str = self.tokenizer.decode([token_id])
                                else:
                                    token_str = self.tokenizer.decode([token_id])
                                
                                if token_str in ['\n', '\0']:
                                    is_finished = True
                            except Exception as e:
                                print(f"Erro ao decodificar token: {e}")
                                is_finished = False
                        candidates.append((new_seq, new_score, is_finished))

                # Selecionar os melhores beam_width candidatos por score
                if not candidates:
                    break
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]

                # Parar cedo se todos os beams estiverem finalizados
                if all(finished for (_, _, finished) in beams):
                    break

        # Retornar a melhor sequência encontrada
        best_seq = max(beams, key=lambda x: x[1])[0]
        try:
            if hasattr(self.tokenizer, 'decode'):
                return self.tokenizer.decode(best_seq)
            else:
                return self.tokenizer.decode(best_seq)
        except Exception as e:
            print(f"Erro ao decodificar a sequência: {e}")
            return ""

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

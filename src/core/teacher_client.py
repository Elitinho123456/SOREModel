"""
teacher_client.py
Abstrai o acesso a modelos Teacher para distilacao de conhecimento.

Providers suportados:
  - OpenAI / OpenAI-compatible (vLLM, LM Studio, etc.)
  - Google Gemini API
  - Ollama (modelos locais: Gemma 4, LLaMA 3, Mistral, etc.)

Qual modo de distilacao usar com cada provider:
  +------------------+--------------------+---------------------------+
  | Provider         | get_logits()       | Modo recomendado          |
  +------------------+--------------------+---------------------------+
  | OpenAI API       | None (nao suporta) | 'sequence'                |
  | Gemini API       | None (nao suporta) | 'sequence'                |
  | Ollama (local)   | None*              | 'sequence' ou 'hybrid'    |
  | vLLM (local)     | Disponivel         | 'logit'                   |
  +------------------+--------------------+---------------------------+

  * O Ollama nao expoe logits brutos via API REST. Para logits reais com
    modelos locais, use vLLM com --api-key e endpoint OpenAI-compatible.
    O OllamaTeacherClient usa 'sequence' por padrao — o teacher gera texto
    de alta qualidade (Gemma 4) e o student aprende a imitar esse texto.
"""
import os
import json
import time
import logging
import requests
from abc import ABC, abstractmethod
from typing import Optional, List

log = logging.getLogger("AZR.TeacherClient")

# Timeout padrao para chamadas ao teacher
_DEFAULT_TIMEOUT = 120  # Ollama com modelos grandes pode demorar


# ─────────────────────────────────────────────────────────────────────────────
#  Interface Base
# ─────────────────────────────────────────────────────────────────────────────

class TeacherClient(ABC):
    """
    Interface base para todos os providers de modelos teacher.

    Todo provider deve implementar:
      generate()   — gera texto dado um prompt (obrigatorio)
      get_logits() — retorna logits brutos (ou None se indisponivel)
    """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Gera texto a partir de um prompt. Retorna string vazia em caso de erro."""
        ...

    @abstractmethod
    def get_logits(self, prompt: str) -> Optional[List[float]]:
        """
        Retorna logits brutos do ultimo token, se disponivel.
        Retorna None para providers que nao expõem logits (API externa).
        """
        ...

    def health_check(self) -> bool:
        """Verifica se o servidor do provider esta acessivel. Override opcional."""
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  OpenAI / OpenAI-compatible (vLLM, LM Studio, etc.)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAITeacherClient(TeacherClient):
    """
    Teacher via API OpenAI (ou qualquer endpoint OpenAI-compatible).
    Suporta vLLM, LM Studio, LocalAI.

    Para suporte a logits reais (modo 'logit' da distilacao):
      Use vLLM com --api-key e a opcao logprobs=True no payload.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        endpoint: str = "https://api.openai.com/v1",
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str = "",
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(
                f"{self.endpoint}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.error(f"[OpenAITeacherClient] generate() falhou: {e}")
            return ""

    def get_logits(self, prompt: str) -> Optional[List[float]]:
        # A API publica da OpenAI nao retorna logits brutos.
        # Para logits via vLLM, seria necessario usar logprobs=True e
        # mapear de volta ao espaco do vocabulario — complexo demais sem
        # acesso ao tokenizador do teacher. Retornamos None por ora.
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Google Gemini API
# ─────────────────────────────────────────────────────────────────────────────

class GeminiTeacherClient(TeacherClient):
    """Teacher via Google Gemini API (cloud).

    Suporte a chave gratuita:
      - Limite: ~15 req/min, 1500 req/dia (gemini-2.0-flash)
      - Ao receber 429 (rate limit), faz retry automatico com backoff exponencial.
      - max_retries=5 com backoff de 2^attempt * base_delay (padrao: ate ~4 min).
    """

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = 5,
        base_delay: float = 15.0,  # segundos — compativel com limite de 15 rpm
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._url = f"{self._BASE}/{model_name}:generateContent"

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str = "",
    ) -> str:
        payload: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self._url}?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )

                # Rate limit (quota gratuita esgotada temporariamente)
                if resp.status_code == 429:
                    wait = self.base_delay * (2 ** attempt)
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        wait = max(wait, float(retry_after))
                    log.warning(
                        f"[GeminiTeacherClient] 429 Rate Limit — aguardando {wait:.0f}s "
                        f"(tentativa {attempt + 1}/{self.max_retries})..."
                    )
                    time.sleep(wait)
                    continue

                # Quota diaria esgotada — nao adianta tentar de novo agora
                if resp.status_code == 403:
                    log.error(
                        "[GeminiTeacherClient] 403 Forbidden — quota diaria possivelmente esgotada. "
                        "Verifique em: https://aistudio.google.com/u/0/plan_information"
                    )
                    return ""

                resp.raise_for_status()
                data = resp.json()
                cands = data.get("candidates", [])
                if cands and "content" in cands[0]:
                    return cands[0]["content"]["parts"][0]["text"]
                return ""

            except requests.Timeout:
                wait = self.base_delay * (2 ** attempt)
                log.warning(
                    f"[GeminiTeacherClient] Timeout ({self.timeout}s) — aguardando {wait:.0f}s "
                    f"(tentativa {attempt + 1}/{self.max_retries})..."
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
                else:
                    log.error("[GeminiTeacherClient] Maximo de tentativas atingido apos timeouts.")
                    return ""

            except requests.ConnectionError as e:
                log.error(f"[GeminiTeacherClient] Erro de conexao: {e}")
                return ""

            except Exception as e:
                log.error(f"[GeminiTeacherClient] generate() falhou: {e}")
                return ""

        log.error("[GeminiTeacherClient] Maximo de tentativas atingido (rate limit persistente).")
        return ""

    def get_logits(self, prompt: str) -> Optional[List[float]]:
        return None  # Gemini API nao expoe logits brutos


# ─────────────────────────────────────────────────────────────────────────────
#  Ollama (modelos locais — Gemma 4, LLaMA 3, Mistral, etc.)
# ─────────────────────────────────────────────────────────────────────────────

class OllamaTeacherClient(TeacherClient):
    """
    Teacher via Ollama (servidor local de modelos open-source).

    Ideal para usar o Gemma 4 como teacher local sem custo de API.

    Instalacao rapida:
        # No servidor Linux com GPU NVIDIA:
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull gemma3:27b       # recomendado — 27B e muito capaz
        ollama pull gemma3:12b       # alternativa mais leve
        ollama pull gemma3:4b        # para maquinas com menos VRAM

    Uso:
        client = OllamaTeacherClient(model_name="gemma3:27b")
        text = client.generate("Explique attention mechanisms")

    Endpoints Ollama usados:
        POST /api/chat    — geracao com historico de mensagens (mais robusto)
        GET  /api/tags    — listar modelos disponiveis (health check)

    Por que /api/chat e nao /api/generate?
        O /api/chat suporta system prompt nativo, e o Gemma 4 foi treinado
        com chat templates — usar o template correto melhora muito a qualidade
        das respostas de distilacao.

    Thinking mode (Gemma 4 / QwQ):
        Modelos como gemma3 e qwq suportam "think before answer".
        Ative com thinking=True para respostas mais raciocioadas
        (importante para distilacao de tarefas complexas).
        O token <think>...</think> e automaticamente filtrado do output.
    """

    def __init__(
        self,
        model_name: str = "gemma3:27b",
        base_url: str = "http://localhost:11434",
        timeout: int = _DEFAULT_TIMEOUT,
        keep_alive: str = "10m",
        context_length: int = 4096,
    ):
        """
        Args:
            model_name:     Nome do modelo no Ollama (ex: "gemma3:27b", "llama3.3:70b").
            base_url:       URL do servidor Ollama. Padrao: localhost:11434.
            timeout:        Timeout em segundos para a requisicao HTTP.
            keep_alive:     Quanto tempo manter o modelo em VRAM apos a ultima requisicao.
                            "10m" = 10 minutos. Use "0" para descarregar imediatamente.
            context_length: Comprimento de contexto (num_ctx). Aumente para prompts longos.
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.context_length = context_length

    def health_check(self) -> bool:
        """Verifica se o Ollama esta rodando e o modelo esta disponivel."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(m.startswith(self.model_name.split(":")[0]) for m in models)
            if not available:
                log.warning(
                    f"[OllamaTeacherClient] Modelo '{self.model_name}' nao encontrado. "
                    f"Disponiveis: {models}. Execute: ollama pull {self.model_name}"
                )
            return available
        except requests.ConnectionError:
            log.error(
                f"[OllamaTeacherClient] Ollama nao acessivel em '{self.base_url}'. "
                "Execute: ollama serve"
            )
            return False
        except Exception as e:
            log.error(f"[OllamaTeacherClient] health_check falhou: {e}")
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str = "",
        thinking: bool = False,
    ) -> str:
        """
        Gera texto usando o modelo Ollama via /api/chat.

        Args:
            prompt:         O prompt do usuario.
            temperature:    Temperatura de amostragem (0.0 = deterministico).
            max_tokens:     Numero maximo de tokens a gerar.
            system_prompt:  Instrucao de sistema (ex: "Responda de forma concisa").
            thinking:       Se True, ativa o thinking mode (Gemma 4, QwQ, etc.).
                            O bloco <think>...</think> e filtrado do resultado final.

        Returns:
            Texto gerado (string). Retorna "" em caso de erro.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": thinking,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.context_length,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            content: str = data.get("message", {}).get("content", "")

            # Filtra o bloco de thinking se presente
            # (Gemma 4 e outros modelos thinking incluem <think>...</think>)
            if thinking and "<think>" in content and "</think>" in content:
                start = content.find("</think>")
                content = content[start + len("</think>"):].strip()

            # Metadados de performance para telemetria do AZR
            eval_count = data.get("eval_count", 0)
            eval_duration_ns = data.get("eval_duration", 1)
            tokens_per_sec = round(eval_count / (eval_duration_ns / 1e9), 1) if eval_count else 0
            log.debug(
                f"[OllamaTeacherClient] {self.model_name} | "
                f"{eval_count} tokens | {tokens_per_sec} tok/s"
            )

            return content

        except requests.ConnectionError:
            log.error(f"[OllamaTeacherClient] Falha de conexao com '{self.base_url}'. Ollama esta rodando?")
            return ""
        except requests.Timeout:
            log.error(f"[OllamaTeacherClient] Timeout ({self.timeout}s). Considere aumentar timeout ou usar modelo menor.")
            return ""
        except Exception as e:
            log.error(f"[OllamaTeacherClient] generate() falhou: {e}")
            return ""

    def get_logits(self, prompt: str) -> Optional[List[float]]:
        """
        O Ollama nao expoe logits brutos via API REST.

        Para logits reais com modelos locais, use vLLM:
            pip install vllm
            vllm serve google/gemma-3-27b-it --api-key token-abc

        Retorna None para usar modo 'sequence' ou 'hybrid'.
        """
        return None

    def list_models(self) -> List[str]:
        """Lista todos os modelos disponiveis no servidor Ollama local."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as e:
            log.error(f"[OllamaTeacherClient] list_models() falhou: {e}")
            return []

    def pull_model(self) -> bool:
        """
        Faz pull do modelo se nao estiver disponivel (equivalente a 'ollama pull').
        Util para automatizar o setup em novas maquinas.
        """
        log.info(f"[OllamaTeacherClient] Fazendo pull do modelo '{self.model_name}'...")
        try:
            resp = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name, "stream": False},
                timeout=600,  # pull pode demorar muito
            )
            resp.raise_for_status()
            status = resp.json().get("status", "")
            log.info(f"[OllamaTeacherClient] Pull concluido: {status}")
            return "success" in status.lower() or status == "pulled successfully"
        except Exception as e:
            log.error(f"[OllamaTeacherClient] pull_model() falhou: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_teacher_client(provider: str, **kwargs) -> TeacherClient:
    """
    Factory para instanciar um TeacherClient pelo nome do provider.

    Args:
        provider: "openai" | "gemini" | "ollama"
        **kwargs: Parametros especificos do provider.

    Exemplos:
        # OpenAI
        client = get_teacher_client("openai", model_name="gpt-4o")

        # Gemini
        client = get_teacher_client("gemini", model_name="gemini-2.0-flash")

        # Ollama local com Gemma 4
        client = get_teacher_client("ollama", model_name="gemma3:27b")

        # Ollama em servidor remoto
        client = get_teacher_client(
            "ollama",
            model_name="gemma3:27b",
            base_url="http://192.168.1.100:11434"
        )
    """
    p = provider.lower().strip()

    if p == "openai":
        return OpenAITeacherClient(
            api_key=kwargs.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            model_name=kwargs.get("model_name", "gpt-4o-mini"),
            endpoint=kwargs.get("endpoint", "https://api.openai.com/v1"),
            timeout=kwargs.get("timeout", _DEFAULT_TIMEOUT),
        )

    elif p == "gemini":
        return GeminiTeacherClient(
            api_key=kwargs.get("api_key", os.environ.get("GEMINI_API_KEY", "")),
            model_name=kwargs.get("model_name", "gemini-2.0-flash"),
            timeout=kwargs.get("timeout", _DEFAULT_TIMEOUT),
        )

    elif p == "ollama":
        return OllamaTeacherClient(
            model_name=kwargs.get("model_name", "gemma3:27b"),
            base_url=kwargs.get("base_url", os.environ.get("OLLAMA_HOST", "http://localhost:11434")),
            timeout=kwargs.get("timeout", _DEFAULT_TIMEOUT),
            keep_alive=kwargs.get("keep_alive", "10m"),
            context_length=kwargs.get("context_length", 4096),
        )

    else:
        raise ValueError(
            f"Provider desconhecido: '{provider}'. "
            f"Opcoes validas: 'openai', 'gemini', 'ollama'."
        )


"""
teacher_client.py
Abstracts access to Teacher models for distillation.
Supports:
- OpenAI API (and compatible endpoints like vLLM, Ollama)
- Google Gemini API
"""
import os
import requests
import json
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union

class TeacherClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def get_logits(self, prompt: str) -> Optional[List[float]]:
        pass

class OpenAITeacherClient(TeacherClient):
    def __init__(self, endpoint: str, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        # Standard chat completions endpoint assumption
        url = f"{self.endpoint}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            print(f"[OpenAITeacherClient] Error: {e}")
            return ""

    def get_logits(self, prompt: str) -> Optional[List[float]]:
        # Most standard APIs don't return full logits easily without specifics (like logprobs=True)
        # This is a placeholder or requires specific endpoint support (like vLLM)
        return None

class GeminiTeacherClient(TeacherClient):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self.api_key = api_key
        self.model_name = model_name
        # Base URL for Gemini
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        url = f"{self.base_url}?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            # Safety checks might block content, need careful parsing
            if 'candidates' in data and data['candidates']:
                candidate = data['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    return candidate['content']['parts'][0]['text']
            return ""
        except Exception as e:
            print(f"[GeminiTeacherClient] Error: {e}")
            return ""

    def get_logits(self, prompt: str) -> Optional[List[float]]:
        # Gemini API currently focuses on text generation
        return None

def get_teacher_client(provider: str, **kwargs) -> TeacherClient:
    if provider.lower() == "openai":
        return OpenAITeacherClient(
            endpoint=kwargs.get("endpoint", "https://api.openai.com/v1"),
            api_key=kwargs.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            model_name=kwargs.get("model_name", "gpt-3.5-turbo")
        )
    elif provider.lower() == "gemini":
        return GeminiTeacherClient(
            api_key=kwargs.get("api_key", os.environ.get("GEMINI_API_KEY", "")),
            model_name=kwargs.get("model_name", "gemini-pro")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

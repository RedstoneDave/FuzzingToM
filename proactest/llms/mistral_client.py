from __future__ import annotations
import os
from typing import List, Dict, Any
from .base import LLMClient, Message

class MistralClient(LLMClient):
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        try:
            from mistralai import Mistral
        except Exception as e:
            raise RuntimeError("mistralai package not installed") from e
        self.client = Mistral(api_key=self.api_key)

    def chat(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        resp = self.client.chat.complete(model=self.model, messages=messages, temperature=kwargs.get("temperature", 0.2))
        text = resp.choices[0].message.content if resp and resp.choices else ""
        return {"text": text, "usage": {}}

from __future__ import annotations
import os
from typing import List, Dict, Any
from .base import LLMClient, Message

class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai package not installed") from e
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.base_url else OpenAI(api_key=self.api_key)

    def chat(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(model=self.model, messages=messages, temperature=kwargs.get("temperature", 0.2))
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return {"text": text, "usage": usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})}

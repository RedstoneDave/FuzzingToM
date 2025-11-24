from __future__ import annotations
import os
from typing import List, Dict, Any
from .base import LLMClient, Message

class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        try:
            import anthropic
        except Exception as e:
            raise RuntimeError("anthropic package not installed") from e
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def chat(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        sys = ""
        user_parts = []
        for m in messages:
            if m["role"] == "system": sys += m["content"] + "\n"
            elif m["role"] == "user": user_parts.append(m["content"])
            elif m["role"] == "assistant": user_parts.append(f"Assistant: {m['content']}")
        content = "\n".join(user_parts)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 512),
            system=sys or None,
            messages=[{"role": "user", "content": content}],
            temperature=kwargs.get("temperature", 0.2),
        )
        text = ""
        try:
            text = "".join(block.text for block in msg.content if hasattr(block, "text"))
        except Exception:
            text = str(msg.content)
        return {"text": text, "usage": {}}

from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any
from .base import BaseAgent
from ..llms.base import Message
from ..llms.openai_client import OpenAIClient
from ..llms.anthropic_client import AnthropicClient
from ..llms.mistral_client import MistralClient
from ..llms.openai_compatible import OpenAICompatibleClient

SYSTEM_PROMPT = "You are a proactive assistant. Given the conversation and unresolved slots, decide whether to ask. If you ask, produce ONE concise question that targets those slots explicitly and avoids redundancy. Return JSON: {\"b\": 0 or 1, \"a\": \"question or empty\"}."

class LLMAgent(BaseAgent):
    def __init__(self, provider: str, model: str, temperature: float = 0.2):
        prov = provider.lower()
        if prov == "openai": self.client = OpenAIClient(model=model)
        elif prov == "anthropic": self.client = AnthropicClient(model=model)
        elif prov == "mistral": self.client = MistralClient(model=model)
        elif prov in ("openai_compat","openai-compatible","compat"): self.client = OpenAICompatibleClient(model=model)
        else: raise ValueError(f"Unknown provider: {provider}")
        self.temperature = temperature

    def decide_and_ask(self, tc: Dict[str, Any]) -> Tuple[int, Optional[str]]:
        convo_text = str(tc.get("conversation", ""))
        unresolved_items = tc.get("unresolved_items", [])
        unresolved = ", ".join(unresolved_items)
        user = f"Conversation so far:\n{convo_text}\n\nUnresolved slots: {unresolved}\nReturn JSON only."
        messages: List[Message] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        out = self.client.chat(messages, temperature=self.temperature)
        text = out.get("text", "").strip()
        import json
        try:
            obj = json.loads(text)
            b = int(obj.get("b", 0))
            a = (obj.get("a") or None) if b else None
            return (1 if b else 0), a
        except Exception:
            if unresolved_items:
                return 1, "Could you clarify: " + "; ".join(unresolved_items) + "?"
            return 0, None

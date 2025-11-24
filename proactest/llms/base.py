from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any

Message = Dict[str, str]

class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> Dict[str, Any]: ...

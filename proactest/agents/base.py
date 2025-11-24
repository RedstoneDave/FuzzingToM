from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

class BaseAgent(ABC):
    @abstractmethod
    def decide_and_ask(self, tc: Dict[str, Any]) -> Tuple[int, Optional[str]]: ...

class ReactiveNeverAsk(BaseAgent):
    def decide_and_ask(self, tc: Dict[str, Any]) -> Tuple[int, Optional[str]]:
        return 0, None

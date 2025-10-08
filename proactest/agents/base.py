from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from ..types import TestCase

class BaseAgent(ABC):
    @abstractmethod
    def decide_and_ask(self, tc: TestCase) -> Tuple[int, Optional[str]]: ...

class ReactiveNeverAsk(BaseAgent):
    def decide_and_ask(self, tc: TestCase) -> Tuple[int, Optional[str]]:
        return 0, None

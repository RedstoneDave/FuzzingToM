from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict

RoleText = Dict[str, str]

@dataclass
class TestCase:
    conversation_id: str
    turn_index: int
    conversation: List[RoleText] | str
    unresolved_items: List[str]
    resolved_items: Dict[str, str] = field(default_factory=dict)
    slot_keywords: Dict[str, List[str]] = field(default_factory=dict)

    def last_user_utterance(self) -> str:
        if isinstance(self.conversation, list):
            for rt in reversed(self.conversation):
                if rt.get("role") == "user":
                    return rt.get("text", "")
            return ""
        return str(self.conversation)

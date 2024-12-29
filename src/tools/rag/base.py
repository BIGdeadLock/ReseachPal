
from abc import ABC, abstractmethod
from typing import Any


from src.domain.state import PaperRagGraphState


class RAGStep(ABC):
    def __init__(self, mock: bool = False) -> None:
        self._mock = mock

    @abstractmethod
    async def generate(self, query: PaperRagGraphState, *args, **kwargs) -> Any:
        pass


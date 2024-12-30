
from abc import ABC, abstractmethod
from typing import Any


from src.domain.queries import Query


class RAGStep(ABC):

    @abstractmethod
    async def agenerate(self, query: Query, *args, **kwargs) -> Any:
        pass

    async def generate(self, query: Query, *args, **kwargs) -> Any:
        pass



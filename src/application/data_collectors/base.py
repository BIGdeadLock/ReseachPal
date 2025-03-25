from abc import ABC, abstractmethod

from src.domain.queries import CollectorQuery
from src.domain.document import Document


class BaseCollector(ABC):
    platform: str

    @abstractmethod
    def collect(self, query: CollectorQuery, **kwargs) -> list[Document]: ...

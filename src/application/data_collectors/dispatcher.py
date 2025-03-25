from loguru import logger

from .arxiv import ArxivCollector
from .base import BaseCollector
from .github import GithubCollector


class DataCollectorDispatcher:
    def __init__(self, mock=False) -> None:
        self._collectors = {}
        self.mock = mock

    @property
    def number_of_collectors(self) -> int:
        return len(self._collectors)

    @classmethod
    def build(cls, mock=False) -> "DataCollectorDispatcher":
        dispatcher = cls(mock)

        return dispatcher

    def register_github(self) -> "DataCollectorDispatcher":
        self.register("github", GithubCollector)
        return self

    def register_arxiv(self) -> "DataCollectorDispatcher":
        self.register("arxiv", ArxivCollector)
        return self

    def register(self, platform: str, collector: type[BaseCollector]) -> None:
        self._collectors[platform] = collector

    def get_collector(self, platform: str) -> BaseCollector:
        for key, collector in self._collectors.items():
            if key.lower() == platform.lower():
                return collector(mock=self.mock)
        else:
            logger.warning(f"No collector found for {platform}")

            return None

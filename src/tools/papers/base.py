from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Union, List
from pydantic import BaseModel

from arxiv import Client


@dataclass
class Paper:
    content: str
    title: str
    published: datetime
    url: str
    relevant_score: float

    def __repr__(self):
        return f"""
        *Title*: {self.title}
        *Published*: {self.published}
        **Summary**: {self.content}
        **URL**: {self.url}
        """


class PaperScorer(BaseModel, ABC):

    @abstractmethod
    def score(self, query: Union[str, List[str]], paper: str) -> float:
        pass

    @abstractmethod
    async def ascore(self, query: Union[str, List[str]], paper: str) -> float:
        pass


@dataclass
class ResearchDependencies:
    client: Client
    paper_scorer: PaperScorer
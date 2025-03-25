from abc import ABC, abstractmethod

from pydantic import BaseModel
from tavily import TavilyClient

from src.config import settings


class WebDocument(BaseModel):
    link: str
    content: str


class WebSearchAdapter(ABC):
    @abstractmethod
    def search(self, query: str) -> list[WebDocument]:
        pass


class TaviliyAdapter(WebSearchAdapter):
    def search(self, query: str) -> list[WebDocument]:
        result = TavilyClient(api_key=settings.TAVILY_API_KEY).search(
            query, max_results=settings.DATA_SOURCE_MAX_RESULTS
        )
        documents = result["results"]
        return [WebDocument(link=document["url"], content=document["content"])
                for document in documents]

from typing import List

from pydantic import BaseModel, Field

from src.domain.types import DataCategory


class Query(BaseModel):

    content: str
    metadata: dict = Field(default_factory=dict)

    class Config:
        category = DataCategory.QUERIES

    @classmethod
    def from_str(cls, query: str) -> "Query":
        return Query(content=query.strip("\n "))

    @classmethod
    def from_kw(cls, kw: List[str]) -> "Query":
        return Query(content=", ".join(kw))

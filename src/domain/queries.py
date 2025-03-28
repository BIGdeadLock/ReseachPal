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

class CollectorQuery(BaseModel):
    content: str
    platform: str | None = None
    rejected_previously: bool = False

    def replace_content(self, new_content: str) -> "CollectorQuery":
        return CollectorQuery(content=new_content, platform=self.platform)
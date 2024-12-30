from pydantic import BaseModel, Field
from typing import List, Union
from typing import Annotated
import operator

from src.domain.document import Paper
from src.domain.queries import Query

class Query(BaseModel):
    query: Union[Query, List[Query]] | None = None
    keywords: List[str] = Field(default_factory=list, description="The keywords to search papers for")
    papers: Annotated[List[Paper], operator.add] = Field(default_factory=list, description="List of papers fetched")

class PaperRagGraphState(BaseModel):
    query: Query
    # papers: List[Paper] = None
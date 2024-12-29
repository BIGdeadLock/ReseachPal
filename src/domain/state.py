from pydantic import BaseModel, Field
from typing import List, Union
from typing import Annotated
import operator

from src.domain.document import Paper
from src.domain.queries import Query

class OverallState(BaseModel):
    query: Union[Query, List[Query]]
    papers: Annotated[List[Paper], operator.add] = Field(default_factory=list, description="List of papers fetched")

class PaperGraphState(BaseModel):
    query: Query
    paper: Paper = None
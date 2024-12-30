from typing import Any, Union, List
from langgraph.types import Send

from src.agent.base import DecisionEdge
from src.agent.graph import QueryExpansion
from src.tools.rag.papers.arxiv_retriever import ArxivPapersRetriever
from src.domain.state import Query

class PapersMapReduce(DecisionEdge):
    start_node_name = QueryExpansion.name
    target_nodes_names = [ArxivPapersRetriever.name]

    def _condition(self, state: Any) -> Union[str, List]:
        return [Send(ArxivPapersRetriever.name, Query(query=q)) for q in state.query]

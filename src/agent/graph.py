
from langgraph.graph import StateGraph, START, END

from src.domain.state import Query
from src.tools.rag.papers.arxiv_retriever import ArxivPapersRetriever
from src.tools.rag.papers.query_expansion import QueryExpansion
from src.agent.cond_edges import PapersMapReduce

graph = StateGraph(Query)

# # --------------------
# # NODES
# #----------------------
graph.add_node(QueryExpansion.name, QueryExpansion().generate)
graph.add_node(ArxivPapersRetriever.name, ArxivPapersRetriever().agenerate)

# # --------------------
# # EDGRES
# #----------------------
graph.add_edge(START, QueryExpansion.name)
graph.add_conditional_edges(*PapersMapReduce().get_decision_params())
graph.add_edge(ArxivPapersRetriever.name, END)

agent = graph.compile()


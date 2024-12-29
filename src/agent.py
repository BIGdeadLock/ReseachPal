
from langgraph.graph import START, StateGraph, END

from src.tools.papers.arxiv import aretrieve_papers
from src.tools.papers.scorer import score_papers



papers_graph = StateGraph(State)

# --------------------
# NODES
#----------------------
papers_graph.add_node(aretrieve_papers)
papers_graph.add_node(score_papers)

# --------------------
# Nodes
#----------------------
papers_graph.ad

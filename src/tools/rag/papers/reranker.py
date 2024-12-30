from typing import List, Any
from loguru import logger
import opik

from src.domain.document import Paper
from src.domain.state import Query
from src.models.embeddings import CrossEncoderModelSingleton

from src.tools.rag.base import RAGStep


class Reranker(RAGStep):

    def __init__(self) -> None:

        super().__init__()
        self._model = CrossEncoderModelSingleton()

    async def agenerate(self, query: Query, *args, **kwargs) -> Any:
        pass

    @opik.track(name="Reranker.generate")
    def generate(self, query: Query, papers: List[Paper], keep_top_k: int) -> List[Paper]:
        documents = [p.content for p in papers]

        query_doc_tuples = [(query.content, d) for d in documents]
        scores = self._model(query_doc_tuples)

        scored_query_doc_tuples = list(zip(scores, papers, strict=False))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]

        logger.info(f"Filtered out: {len(reranked_documents)} out of {len(papers)}")
        return reranked_documents
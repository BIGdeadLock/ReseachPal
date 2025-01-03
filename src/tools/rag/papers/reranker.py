from typing import List, Any
from loguru import logger
import opik
import numpy as np
from src.domain.document import Paper
from src.domain.state import Query
from src.models.embeddings import CrossEncoderModelSingleton

from src.tools.rag.base import RAGStep
from sklearn.metrics.pairwise import cosine_similarity

class MMRFilter(RAGStep):


    def __init__(self):
        super().__init__()

    async def agenerate(self, query: Query, papers: List[Paper], threshold: 0.5) -> List[Paper]:

        papers_emb = [paper.embedding for paper in papers]
        idx_and_scores = self._maximal_marginal_relevance(query.embedding, papers_emb)

        for idx, score in enumerate(idx_and_scores):
            papers[idx].relevant_score = score

        filtered_papers = list(filter(lambda p: p.relevant_score >= threshold, papers))
        return sorted(filtered_papers, key=lambda x: x.relevant_score, reverse=True)



    def _maximal_marginal_relevance(self,query_embedding: list, embedding_list: list, lambda_mult: float = 0.5, k: int = 4) -> List[int]:
        """Calculate maximal marginal relevance."""
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        if min(k, len(embedding_list)) <= 0:
            return []
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
        most_similar = int(np.argmax(similarity_to_query))
        scores = [np.max(similarity_to_query) * lambda_mult]
        idxs = [most_similar]
        selected = np.array([embedding_list[most_similar]])
        while len(idxs) < min(k, len(embedding_list)):
            best_score = -np.inf
            idx_to_add = -1
            similarity_to_selected = cosine_similarity(embedding_list, selected)
            for i, query_score in enumerate(similarity_to_query):
                if i in idxs:
                    continue
                redundant_score = max(similarity_to_selected[i])
                equation_score = (
                    lambda_mult * query_score - (1 - lambda_mult) * redundant_score
                )
                if equation_score > best_score:
                    best_score = equation_score
                    idx_to_add = i
            idxs.append(idx_to_add)
            scores.append(equation_score)
            selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)

        result = list(zip(idxs, scores))
        return result


class Reranker(RAGStep):
    """
    Reranker class that inherits from RAGStep and is responsible for reranking a list of papers based on their relevance to a given query.

    Attributes:
        _model (CrossEncoderModelSingleton): An instance of CrossEncoderModelSingleton used for scoring the relevance of documents.
    """

    def __init__(self) -> None:

        super().__init__()
        self._cross_enc_model = CrossEncoderModelSingleton()

    @opik.track(name="Reranker.generate", capture_input=False, capture_output=True)
    def generate(self, query: Query, papers: List[Paper], keep_top_k: int) -> List[Paper]:
        """
        Generates a reranked list of papers based on their relevance to the given query.

        Args:
            query (Query): The query object containing the search query.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the generation process.
        """
        pass

    @opik.track(name="Reranker.agenerate", capture_input=False, capture_output=True)
    async def agenerate(self, query: Query, papers: List[Paper], keep_top_k: int) -> List[Paper]:
        """
        Generates a reranked list of papers based on their relevance to the given query.

        Args:
            query (Query): The query object containing the search query.
            papers (List[Paper]): A list of Paper objects to be reranked.
            keep_top_k (int): The number of top relevant papers to keep after reranking.
            mmr_threshold (int, optional): MMR threshold used to filter similar documents before reranking. Defaults to 0.

        Returns:
            List[Paper]: A list of the top `keep_top_k` most relevant Paper objects.
        """
        documents = [p.content for p in papers]

        query_doc_tuples = [(query.content, d) for d in documents]
        scores = self._cross_enc_model(query_doc_tuples)

        scored_query_doc_tuples = list(zip(scores, papers, strict=False))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)

        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]

        logger.info(f"Filtered out: {len(reranked_documents)} out of {len(papers)}")
        return reranked_documents
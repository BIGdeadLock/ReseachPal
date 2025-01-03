import asyncio
import concurrent.futures
from typing import List
from loguru import logger
from arxiv import Search, Client
from pydantic import PrivateAttr

from src.domain.document import Paper
from src.domain.queries import Query
from src.tools.rag.base import RAGStep
from src.tools.rag.papers.reranker import Reranker
from src.tools.rag.papers.query_expansion import QueryExpansion
from src.tools.rag.papers.scorer import LlmAsJudge
from src.utils.config import config
from src.monitoring.opik import configure_opik
from src.utils import misc

class PapersRetriever(RAGStep):
    name = "arxiv_retriever"
    _client: Client = PrivateAttr()
    _query_expander = PrivateAttr()
    _reranker: Reranker = PrivateAttr()
    _scorer: LlmAsJudge = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._query_expander = QueryExpansion()
        self._client = Client()
        self._reranker = Reranker()
        self._scorer = LlmAsJudge()


    async def agenerate(self, query: Query, **kwargs) -> List[Paper]:
        """Returns the most relevant papers with relevance scores based on the user query
        Two steps are applied: First - MMR score for diversity, then Cross encoder for relevance

            Args:
                search_queries: Can be a list of queries or keywords: [q1, q2,...]. It can also be a string.

            Returns: A list of Papers.
            :param **kwargs:
            :param **kwargs:
            """

        n_generated_queries = await QueryExpansion().agenerate(query)

        logger.info(f"Successfully generated {len(n_generated_queries)} search queries.",)

        # Use asyncio to fetch papers for each query concurrently
        search_tasks = [
            self._search_and_rerank(_query_model, config.arxiv.max_results, config.arxiv.keep_top_k_results)
            for _query_model in n_generated_queries
        ]
        all_papers = await asyncio.gather(*search_tasks)

        # Flatten the results and remove duplicates
        n_k_documents = misc.flatten(all_papers)
        n_k_documents = list(set(n_k_documents))

        papers = await self._score(query, n_k_documents)

        return papers

    async def _search_and_rerank(self, query: Query, top_k: int, keep_top_k: int) -> List[Paper]:
        """
        Searches and reranks papers for a given query.

        Args:
            query: The search query.
            top_k: Maximum number of search results.
            keep_top_k: Number of top results to keep after reranking.

        Returns:
            A list of reranked Papers.
        """
        # Search for papers
        search_results = await asyncio.to_thread(self._search, query, top_k)

        # Rerank the search results
        reranked_papers = await self._rerank(query, search_results, keep_top_k)

        return reranked_papers

    def _search(self, query: Query, top_k) -> List[Paper]:
        search = Search(
            query=query.content,
            max_results=top_k,
        )
        return [
            Paper(
                content=paper.summary,
                title=paper.title,
                published=paper.published,
                url=paper.pdf_url,
                relevant_score=0,
            )
            for paper in Client().results(search)
        ]

    async def _rerank(self, query: Query, papers: list[Paper], keep_top_k: int):

        reranked_documents = await self._reranker.agenerate(query=query, papers=papers, keep_top_k=keep_top_k)
        logger.info(f"{len(reranked_documents)} documents reranked successfully.")
        return reranked_documents

    async def _score(self, query: Query, papers: list[Paper]) -> list[Paper]:
        tasks = [
            self._scorer.agenerate(query, paper)
            for paper in papers
        ]
        scores = await asyncio.gather(*tasks)
        for score, paper in zip(scores,papers):
            paper.relevant_score = score

        return papers



if __name__ == '__main__':
    configure_opik()
    query = Query.from_str("Analysis of LLM's role in enhancing cybersecurity awareness and training programs.")
    retriever = PapersRetriever()
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(retriever.agenerate(query))
    for paper in result:
        logger.info(paper)




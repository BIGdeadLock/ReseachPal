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

            Args:
                search_queries: Can be a list of queries or keywords: [q1, q2,...]. It can also be a string.

            Returns: A list of Papers.
            :param **kwargs:
            :param **kwargs:
            """

        n_generated_queries = await self._query_expander.generate(query)
        logger.info(f"Successfully generated {len(n_generated_queries)} search queries.",)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [executor.submit(self._search, _query_model, config.arxiv.max_results)
                            for _query_model in n_generated_queries]

            n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
            n_k_documents = misc.flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))

        logger.info(f"Successfully fetched {len(n_k_documents)} papers.", )

        ranked_documents = self._rerank(query, n_k_documents, config.arxiv.keep_top_k_results)
        papers = await self._score(query, ranked_documents)

        return papers

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

    def _rerank(self, query: Query, papers: list[Paper], keep_top_k: int):

        reranked_documents = self._reranker.generate(query=query, papers=papers, keep_top_k=keep_top_k)
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




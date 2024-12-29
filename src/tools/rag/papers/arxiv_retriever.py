import asyncio
from typing import Any
from loguru import logger
from arxiv import Search, Client
from pydantic import PrivateAttr

from src.domain.document import Paper
from src.domain.queries import Query
from src.domain.state import PaperRagGraphState
from src.tools.rag.base import RAGStep
from src.utils.config import config
from src.tools.rag.papers.scorer import LlmAsJudge
from src.utils.opik_utils import configure_opik


class ArxivPapersRetriever(RAGStep):
    name = "arxiv_retriever"
    _client: Client = PrivateAttr()
    _ranker: LlmAsJudge = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._client = Client()
        self._ranker = LlmAsJudge()

    async def generate(self, state: PaperRagGraphState, *args, **kwargs) -> Any:
        """Returns the most relevant papers with relevance scores based on the user query

            Args:
                search_queries: Can be a list of queries or keywords: [q1, q2,...]. It can also be a string.

            Returns: A list of Papers.
            """
        search = Search(
            query=str(state.query.content),
            max_results=config.arxiv.max_results,
        )
        papers = [
            Paper(
                content=paper.summary,
                title=paper.title,
                published=paper.published,
                url=paper.pdf_url,
                relevant_score=0,
            )
            for paper in Client().results(search)
        ]

        tasks = [
            self._ranker.generate(state.query, paper)
            for paper in papers
        ]
        scores = await asyncio.gather(*tasks)
        for paper, score in zip(papers, scores):
            paper.relevant_score = score

        if config.arxiv.relevance_score_threshold:
            papers = list(filter(lambda p: p.relevant_score >= config.arxiv.relevance_score_threshold, papers))

        return {"papers": papers}


if __name__ == '__main__':
    configure_opik()
    query = Query.from_str("Analysis of LLM's role in enhancing cybersecurity awareness and training programs.")
    state = PaperRagGraphState(query=query)
    retriever = ArxivPapersRetriever()
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(retriever.generate(state))
    for paper in result["papers"]:
        logger.info(paper)




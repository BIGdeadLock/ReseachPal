import asyncio
from datetime import datetime
from typing import List, Union
from arxiv import Search
from pydantic_ai import RunContext

from src.tools.papers.base import Paper
from src.tools.papers.base import ResearchDependencies
from src.utils.config import config



async def aretrieve_papers(context: RunContext[ResearchDependencies], search_queries: Union[str, List[str]]) -> List[Paper]:
    """Returns the most relevant papers with relevance scores based on the user query

    Args:
        search_queries: Can be a list of queries or keywords: [q1, q2,...]. It can also be a string.

    Returns: A list of Papers.
    """
    if type(search_queries) is str:
        search_queries = [search_queries]

    papers = []
    for query in search_queries:
        search = Search(
            query=str(query),
            max_results=config.arxiv.max_results,
        )
        results = [
            Paper(
                content=paper.summary,
                title=paper.title,
                published=paper.published,
                url=paper.pdf_url,
                relevant_score=1,
            )
            for paper in context.deps.client.results(search)]
        tasks = [
            context.deps.paper_scorer.ascore(query, paper.content)
            for paper in results
        ]
        scores = await asyncio.gather(*tasks)
        for paper, score in zip(results, scores):
            paper.relevant_score = score

        papers += results

    papers = list(filter(lambda paper: paper.relevant_score > config.arxiv.relevance_score_threshold, papers))
    if config.arxiv.relevance_score_threshold:
        papers = list(filter(lambda paper: paper.published.year > config.arxiv.publish_year_threshold, papers))

    return papers
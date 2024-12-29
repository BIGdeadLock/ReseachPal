import asyncio
from typing import List, Union
from arxiv import Search, Client


from src.tools.papers.base import Paper
from src.agent import State
from src.utils.config import config



async def aretrieve_papers(state: State):
    """Returns the most relevant papers with relevance scores based on the user query

    Args:
        search_queries: Can be a list of queries or keywords: [q1, q2,...]. It can also be a string.

    Returns: A list of Papers.
    """

    search = Search(
        query=str(state.query),
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
    return {"papers": papers}



from src.domain.document import Report
from src.domain.queries import Query

from src.tools.rag.papers import PapersReportGenerator, PapersRetriever, QueryKwExtraction
from src.utils.config import config

async def get_interesting_papers(query: Query) -> Report:
    query.keywords = await QueryKwExtraction().agenerate(query)
    retriever = PapersRetriever()
    papers = await retriever.agenerate(query)


    if config.arxiv.relevance_score_threshold:
        papers = list(filter(lambda p: p.relevant_score >= config.arxiv.relevance_score_threshold, papers))

    report = await PapersReportGenerator().agenerate(query, papers)
    return report
from arxiv import Client, Search
from loguru import logger

from src.domain.document import Document, DocumentMetadata
from src.domain.queries import CollectorQuery
from src.config import settings

from .base import BaseCollector


class ArxivCollector(BaseCollector):
    platform = "arxiv"

    def __init__(self, mock=False) -> None:
        super().__init__()
        self._mock = mock

    def collect(self, query: CollectorQuery, **kwargs) -> list[Document]:
        logger.info(f"Starting retrieving papers for query: {query.content}")

        search = Search(
            query=query.content.strip(),
            max_results=settings.DATA_SOURCE_MAX_RESULTS,
        )

        count = 0
        documents = []
        for paper in Client().results(search):
            documents.append(
                Document(
                    content=paper.summary,
                    metadata=DocumentMetadata(
                        title=paper.title,
                        url=paper.pdf_url,
                        platform=self.platform,
                        properties=dict(release_date=paper.published.strftime("%Y-%M-%d"),)
                    )
                )
            )
            count += 1

        logger.info(f"Finished retrieving {count} papers")

        return documents
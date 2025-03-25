from gitingest import ingest
from loguru import logger
from sqlalchemy.testing.suite.test_reflection import metadata

from src.utils.misc import num_tokens_from_string
from src.domain.queries import CollectorQuery
from src.models.gemini import Gemini

from .base import BaseCollector
from .constants import MAX_TOKENS_ALLOWED, TAVILIY_GITHUB_MOCK_RESULTS
from .web import TaviliyAdapter, WebDocument
from ...domain.document import Document, DocumentMetadata


class GithubCollector(BaseCollector):
    platform = "github"

    def __init__(self, ignore=None, mock=False) -> None:
        super().__init__()
        self._ignore = ignore or {"*.git", "*.toml", "*.lock", "*.png"}
        self._mock = mock

    def collect(self, query: CollectorQuery, **kwargs) -> list[Document]:
        logger.info(f"Searching online for github repositories that matches the query: {query.content}")
        if self._mock:
            logger.warning("Using Mock Data")
            web_documents = TAVILIY_GITHUB_MOCK_RESULTS
            web_documents = [WebDocument(link=document["url"], content=document["content"])
                for document in web_documents]
        else:
            web_documents = TaviliyAdapter().search(query.content)

        documents = []

        for document in web_documents:
            link = document.link

            if "github" not in link:
                logger.warning(f"Found non github link: {link}, Skipping")
                continue

            logger.info(f"Starting scrapping GitHub repository: {link}")

            # Ingest only the readme file which will contain a summary for the repo
            summary, tree, content = ingest(link, include_patterns="*.md", exclude_patterns=self._ignore)
            readme = content.replace("=", "").replace("\n", " ").strip()

            num_tokens = num_tokens_from_string(readme)
            # Create a small summary for the readme if it is above 500 tokens
            if num_tokens > MAX_TOKENS_ALLOWED:
                logger.warning(
                    f"Github README has {num_tokens}," f" summarizing it to {MAX_TOKENS_ALLOWED} tokens"
                )
                readme = Gemini().generate(query=f"Create a concise summary for the following: {readme}")

            documents.append(
                Document(
                    content=readme,
                    summary=summary,
                    metadata=DocumentMetadata(
                        url=link,
                        title=link.split("/")[-1],
                        platform=self.platform,
                    )
                )
            )


            logger.info(f"Finished scrapping GitHub repository: {link}")

        return documents

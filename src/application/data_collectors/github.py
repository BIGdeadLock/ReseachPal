from gitingest import ingest
from loguru import logger

from src.domain.queries import CollectorQuery
from src.models.gemini import Gemini
from .base import BaseCollector
from .constants import MAX_TOKENS_ALLOWED, TAVILIY_GITHUB_MOCK_RESULTS
from .web import TaviliyAdapter, WebDocument, TavilySearchResultFilter
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

        # Filter results to include only github results
        web_documents = TavilySearchResultFilter().platform_filter(platform=query.platform, results=web_documents)

        documents = []

        for document in web_documents:
            link = document.link

            logger.info(f"Starting scrapping GitHub repository: {link}")

            # Ingest only the readme file which will contain a summary for the repo
            summary, tree, content = ingest(link, include_patterns="*.md", exclude_patterns=self._ignore)
            readme = content.replace("=", "").replace("\n", " ").strip()

            # Create a small summary for the readme if it is above 500 tokens
            readme = Gemini().generate(
                query=f"""
                You will get a github repository project description. 
                Your task is to createa summary for it.
                Constraints:
                - It should not take more than 3 sentences.
                - It should capture the core capabilities that the repository's project provide.
                - It should include the domain or use case the project is aiming for. For example: Prompt Monitor, Cybersecurity Threat Intelligence, Unit tests, etc. 
                - Answer only with the summary.
                - Don't start your answer with 'Summary:', instead just write the summary.
                
                
                Github description: {readme}
                """)

            documents.append(
                Document(
                    content=readme,
                    metadata=DocumentMetadata(
                        url=link,
                        title=link.split("/")[-1],
                        platform=self.platform,
                        properties=dict(query=query.content.strip())
                    )
                )
            )


            logger.info(f"Finished scrapping GitHub repository: {link}")

        return documents



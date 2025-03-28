import opik
from loguru import logger

from src.domain.document import Document
from src.prompts.templates import QueryBuilderPromptTemplate
from src.domain.queries import CollectorQuery
from src.models.gemini import Gemini
from src.infrastructure.mongo import MongoDBService
from src.utils.constants import RAW_DOCUMENT_COLLECTION_NAME, RELEVANT_SCORE_DEFAULT

class QueryBuilder:
    def __init__(self, mock=False):
        self._mock = mock
        self._mongo_client = MongoDBService(model=Document, collection_name=RAW_DOCUMENT_COLLECTION_NAME)


    @opik.track(name="QueryBuilder.generate")
    def generate(self, query: CollectorQuery) -> CollectorQuery:
        if self._mock:
            return CollectorQuery(content="LLM agents in cybersecurity")

        query_expansion_template = QueryBuilderPromptTemplate()

        prompt = query_expansion_template.create_template(fields=query.content, platform=query.platform,
                                                          positives=self.get_positive_documents(query.platform),
                                                          negatives=self.get_negative_documents(query.platform))

        content = Gemini().generate(query=prompt)
        query = query.replace_content(content)

        self._mongo_client.close()
        return query

    def get_positive_documents(self, platform) -> list[Document]:
        ranked_documents = self._mongo_client.fetch_documents(5,
                                                   {
                                                            "user_score": {"$gt": RELEVANT_SCORE_DEFAULT},
                                                            "metadata.platform": platform
                                                        }
                                                    )

        logger.info(f"Fetched {len(ranked_documents)} Positive ranked documents")
        return ranked_documents

    def get_negative_documents(self, platform) -> list[Document]:
        ranked_documents = self._mongo_client.fetch_documents(5,
                                                   {
                                                            "user_score": {"lte": RELEVANT_SCORE_DEFAULT},
                                                            "metadata.platform": platform
                                                        }
                                                    )

        logger.info(f"Fetched {len(ranked_documents)} Negative ranked documents")
        return ranked_documents

if __name__ == "__main__":
    query = CollectorQuery(content=",".join(["LLM", "Agents", "Cybersecurity"]), platform="arxiv")
    query_builder = QueryBuilder()
    query = query_builder.generate(query)
    logger.debug(query)

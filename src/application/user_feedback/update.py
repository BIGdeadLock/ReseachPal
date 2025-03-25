from loguru import logger

from src.application.user_feedback import QualityFilter
from src.domain.document import Document, EmbeddedDocument
from src.infrastructure.mongo import MongoDBService
from src.infrastructure.qdrant import QdrantDatabaseConnector
from src.models import EmbeddingModelSingleton
from src.utils.constants import RAW_DOCUMENT_COLLECTION_NAME, FAVORITES_COLLECTION_NAME, FIND_ONE_URL_QUERY_KEY


def update(document: Document, threshold: int = 3):

    # Step 1 - Update the feedback score
    with MongoDBService(model=Document, collection_name=RAW_DOCUMENT_COLLECTION_NAME) as service:
       service.update_documents(({FIND_ONE_URL_QUERY_KEY: document.metadata.url},
                                dict(user_score=document.user_score)))

    logger.info(f"Updated Document score in {RAW_DOCUMENT_COLLECTION_NAME} collection")

    # Step 2 - Filter the documents to only update quality one
    quality_filter = QualityFilter(threshold=threshold)
    if quality_filter.is_quality_document(document):

        # Step 3 - embed and load
        model = EmbeddingModelSingleton()
        embedding = model([document.content])[0]
        embed_doc = EmbeddedDocument.from_document_embedding(document, embedding)

        with QdrantDatabaseConnector(
                collection_name=FAVORITES_COLLECTION_NAME
        ) as qdrant_client:
            qdrant_client.bulk_insert([embed_doc])

        logger.info(f"Updated Embedded document in Vector DB collection")
        return True

    logger.info(f"Document has low quality score - Skipping feature engineering step")
    return False


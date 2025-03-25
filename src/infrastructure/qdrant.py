from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import exceptions

from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np

from src.config import settings
from src.domain.document import Document
from src.models.embeddings import EmbeddingModelSingleton


class QdrantDatabaseConnector:
    _client: QdrantClient | None = None

    def __init__(self, collection_name: str = "favorites"):
        self._collection_name = collection_name
        try:
            if settings.USE_QDRANT_CLOUD:
                self._client = QdrantClient(
                    url=settings.QDRANT_CLOUD_URL,
                    api_key=settings.QDRANT_APIKEY,
                )

                uri = settings.QDRANT_CLOUD_URL
            else:
                self._client = QdrantClient(
                    host=settings.QDRANT_DATABASE_HOST,
                    port=settings.QDRANT_DATABASE_PORT,
                )

                uri = f"{settings.QDRANT_DATABASE_HOST}:{settings.QDRANT_DATABASE_PORT}"

            logger.info(f"Connection to Qdrant DB with URI successful: {uri}")
        except exceptions.UnexpectedResponse:
            logger.exception(
                "Couldn't connect to Qdrant.",
                host=settings.QDRANT_DATABASE_HOST,
                port=settings.QDRANT_DATABASE_PORT,
                url=settings.QDRANT_CLOUD_URL,
            )

            raise

    def to_point(self,document: Document) -> PointStruct:

        payload = document.metadata.model_dump() | dict(content = document.content)

        _id = str(payload.pop("id"))
        vector = payload.pop("embedding", {})
        if vector and isinstance(vector, np.ndarray):
            vector = vector.tolist()

        return PointStruct(id=_id, vector=vector, payload=payload)

    def bulk_insert(self, documents: list[Document]) -> bool:
        try:
            self._bulk_insert(documents)
        except exceptions.UnexpectedResponse:
            logger.info(
                f"Collection '{self._collection_name}' does not exist. Trying to create the collection and reinsert the documents."
            )

            self._client.create_collection()

            try:
                self._bulk_insert(documents)
            except exceptions.UnexpectedResponse:
                logger.error(f"Failed to insert documents in '{self._collection_name}' collection.")

                return False

        return True

    def create_collection(self) -> bool:
        vectors_config = VectorParams(size=EmbeddingModelSingleton().embedding_size, distance=Distance.COSINE)

        return self._client.create_collection(collection_name=self._collection_name, vectors_config=vectors_config)


    def _bulk_insert(self, documents: list[Document]) -> None:
        points = [self.to_point(doc) for doc in documents]

        self._client.upsert(collection_name=self._collection_name, points=points)

    def close(self):
        self._client.close()
        logger.debug("Closed Qdrant connection.")

    def __enter__(self) -> "QdrantDatabaseConnector":
        """Enable context manager support.

        Returns:
            MongoDBService: The current instance.
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close Qdrant connection when exiting context.

        Args:
            exc_type: Type of exception that occurred, if any.
            exc_val: Exception instance that occurred, if any.
            exc_tb: Traceback of exception that occurred, if any.
        """

        self.close()
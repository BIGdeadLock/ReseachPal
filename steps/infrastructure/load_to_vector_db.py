
from zenml.steps import step, get_step_context

from src.domain.document import Document
from src.infrastructure.qdrant import (
    QdrantDatabaseConnector
)


@step
def load(
    documents: list[Document],
    collection_name: str = "favorites",
) -> bool:
    """Process documents by chunking, embedding, and loading into MongoDB.

    Args:
        documents: List of documents to process.
        collection_name: Name of MongoDB collection to store documents.
        device: Device to run embeddings on ('cpu' or 'cuda'). Defaults to 'cpu'.
    """


    with QdrantDatabaseConnector(
        collection_name=collection_name
    ) as qdrant_client:
        qdrant_client.bulk_insert(documents)

    step_context = get_step_context()
    step_context.add_output_metadata(dict(collection_name=collection_name))

    return True

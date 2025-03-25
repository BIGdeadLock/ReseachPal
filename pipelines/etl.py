
from loguru import logger
from zenml import pipeline

from steps.etl import collect_documents
from steps.infrastructure import (
    ingest_to_mongodb
)

from src.utils.constants import RAW_DOCUMENT_COLLECTION_NAME

@pipeline
def etl(
    interested: list[str], platforms: list[str]
) -> None:
    with logger.contextualize(task="Document collection"):
        documents = collect_documents(interested, platforms)

    with logger.contextualize(task="MongoDB collection Ingest"):
        ingest_to_mongodb(
            models=documents,
            collection_name=RAW_DOCUMENT_COLLECTION_NAME,
            clear_collection=True,
        )

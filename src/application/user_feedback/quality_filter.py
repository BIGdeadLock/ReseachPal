from loguru import logger

from src.domain.document import Document


class QualityFilter:

    def __init__(self, threshold: int):
        self.threshold = threshold

    def filter(self, documents: list[Document] | Document) -> list[Document]:

        if type(documents) is not list:
            documents = [documents]

        logger.info(f"Filtering {len(documents)} documents using threshold {self.threshold}")
        quality_docs = list(filter(lambda doc: doc.quality >= self.threshold, documents))
        logger.info(f"Filtered {len(quality_docs)} documents")

        return quality_docs

    def is_quality_document(self, document: Document) -> bool:
        return document.user_score >= self.threshold

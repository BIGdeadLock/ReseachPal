
from pydantic import BaseModel, Field

from src.utils.misc import generate_random_hex

class DocumentMetadata(BaseModel):
    url: str
    platform: str
    title: str
    properties: dict = Field(default_factory=dict)


class Document(BaseModel):
    id: str = Field(default_factory=lambda: generate_random_hex(length=32))
    metadata: DocumentMetadata
    content: str
    user_score: float | None = None
    summary: str | None = None

    def add_summary(self, summary: str) -> "Document":
        self.summary = summary

        return self

    def add_quality_score(self, score: float) -> "Document":
        self.user_score = score

        return self


    def __eq__(self, other: object) -> bool:
        """Compare two Document objects for equality.

        Args:
            other: Another object to compare with this Document.

        Returns:
            bool: True if the other object is a Document with the same ID.
        """
        if not isinstance(other, Document):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Generate a hash value for the Document.

        Returns:
            int: Hash value based on the document's ID.
        """
        return hash(self.id)


class EmbeddedDocument(Document):
    embedding: list[float]

    @classmethod
    def from_document_embedding(cls, document: Document, embedding: list[float]) -> "EmbeddedDocument":
        return EmbeddedDocument(
            content=document.content,
            metadata=document.metadata,
            embedding=embedding
        )
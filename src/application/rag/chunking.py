from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

from src.models.embeddings import EmbeddingModelSingleton
from src.domain.document import Document

embedding_model = EmbeddingModelSingleton()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 0) -> list[str]:
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0)
    text_split_by_characters = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=embedding_model.max_input_length,
        model_name=embedding_model.model_id,
    )
    chunks_by_tokens = []
    for section in text_split_by_characters:
        chunks_by_tokens.extend(token_splitter.split_text(section))

    return chunks_by_tokens


def chunk_document(document: Document, chunk_size: int = 500, chunk_overlap: int = 0) -> list[Document]:

    chunks = chunk_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return [
        Document(
            content=chunk,
            metadata=document.metadata,
        )
        for chunk in chunks
    ]
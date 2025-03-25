from typing import Annotated

from zenml.steps import step, get_step_context

from src.domain.document import Document, EmbeddedDocument
from src.models import EmbeddingModelSingleton

model = EmbeddingModelSingleton()

@step
def embed(documents: list[Document]) -> Annotated[list[EmbeddedDocument], "embedded_documents"]:
    num_docs = len(documents)
    contents = [doc.content for doc in documents]
    embeddings = model(contents)

    embedded_documents = []
    for doc, embedding in zip(documents, embeddings):
        embedded_documents.append(
            EmbeddedDocument.from_document_embedding(doc, embedding)
        )

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="chunks", metadata={
        "num_docs": num_docs,
        "embedding_size": model.embedding_size,
        "model_id": model.model_id,
    })

    return embedded_documents
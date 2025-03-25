from typing import Annotated

from zenml.steps import step, get_step_context


from src.domain.document import Document
from src.application.rag import chunk_document

@step
def chunk(documents: list[Document]) -> Annotated[list[Document], "chunks"]:

    num_docs = len(documents)
    chunks = []

    for doc in documents:
        chunks.extend(chunk_document(doc, num_docs))

    num_chunks = len(chunks)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="chunks", metadata={
        "num_docs": num_docs,
        "num_chunks": num_chunks,
    })

    return chunks
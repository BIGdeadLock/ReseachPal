from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import get_step_context, step

from src.application.data_collectors.dispatcher import DataCollectorDispatcher
from src.application.data_collectors.query_builder import QueryBuilder
from src.domain.document import Document
from src.domain.queries import CollectorQuery


@step
def collect_documents(
    field_of_interest: list[str], platforms: list[str]
) -> Annotated[list[Document], "documents"]:
    dispatcher = DataCollectorDispatcher.build(mock=True).register_github().register_arxiv()

    logger.info(f"Starting to retrieve documents for {len(field_of_interest)} field of interest.")

    metadata = {}
    successfull_collections = 0
    collected = []
    for platform in tqdm(platforms):
        documents = _collect_document(dispatcher, field_of_interest, platform)
        collected.extend(documents)
        successfull_collections += len(documents)
        metadata[platform] = len(documents)

    metadata['total'] = successfull_collections
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="documents", metadata=metadata)

    logger.info(
        f"Successfully retrieved documents for {successfull_collections} for "
        f"{dispatcher.number_of_collectors} collectors."
    )

    return collected


def _collect_document(
    dispatcher: DataCollectorDispatcher, fields: list[str], platform: str
) -> list[Document]:
    collector = dispatcher.get_collector(platform)
    if collector is None:
        return []

    query_builder = QueryBuilder()
    query = CollectorQuery(content=str(fields), platform=platform)
    query = query_builder.generate(query)

    try:
        return collector.collect(query=query)

    except Exception as e:
        logger.error(f"An error occurred while collecting using {collector.platform} collector: {e!s}")

        return []


def _add_to_metadata(metadata: dict, source: str, successfull_ret: bool) -> dict:
    if source not in metadata:
        metadata[source] = {}
    metadata[source]["successful"] = metadata.get(source, {}).get("successful", 0) + successfull_ret
    metadata[source]["total"] = metadata.get(source, {}).get("total", 0) + 1

    return metadata


if __name__ == "__main__":
    field_of_interest = ["LLM", "Cybersecurity"]
    platforms = ["arxiv", "github"]
    collect_documents(field_of_interest, platforms)

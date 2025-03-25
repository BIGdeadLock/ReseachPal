import opik
from loguru import logger

from src.prompts.templates import QueryBuilderPromptTemplate
from src.domain.queries import CollectorQuery
from src.models.gemini import Gemini


class QueryBuilder:
    def __init__(self, mock=False):
        self._mock = mock

    @opik.track(name="QueryBuilder.generate")
    def generate(self, query: CollectorQuery) -> CollectorQuery:
        if self._mock:
            return CollectorQuery(content="LLM agents in cybersecurity")

        logger.info(f"Creating new query from {query.content} to fetch new content from {query.platform}")
        query_expansion_template = QueryBuilderPromptTemplate()
        prompt = query_expansion_template.create_template(fields=query.content, platform=query.platform)

        content = Gemini().generate(query=prompt)
        query = query.replace_content(content)
        logger.info(f"New query created: {query.content}")
        return query


if __name__ == "__main__":
    query = CollectorQuery(content=",".join(["LLM", "Agents", "Cybersecurity"]), platform="github")
    query_builder = QueryBuilder()
    query = query_builder.generate(query)
    logger.debug(query)

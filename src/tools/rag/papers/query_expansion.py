from typing import List

import opik
from loguru import logger
from langchain_openai.chat_models import AzureChatOpenAI
import asyncio
from openai import BaseModel
from pydantic import PrivateAttr

from src.domain.prompt import Prompt
from src.domain.queries import Query
from src.utils.constants import OPENAI_GPT4O_MINI_DEPLOYMENT_ID as GPT4_MINI, OPENAI_API_GPTO_VERSION as API_VER
from src.utils.config import config
from src.monitoring.opik import configure_opik

prompt = Prompt(
   """
    You are a expert in the cybersecurity domain.
    
    *Task:* Break down the user's query to sub domains the user may be interested in.
    
    Return a list of 3 new queries.
    
    --------------------------------
    *User Query:* {query}
    """
)

class Queries(BaseModel):
    queries: List[str]

class QueryExpansion:
    name = "query_expansion"
    _llm = PrivateAttr()

    def __init__(self):
        self._llm = AzureChatOpenAI(
            azure_endpoint=config.openai.url,
            openai_api_version=API_VER,
            openai_api_key=config.openai.api_key,
            deployment_name=GPT4_MINI,
            temperature=0
        )

    @opik.track(name="QueryExpansion.generate")
    async def generate(self, query: Query) -> List[Query]:
        llm = self._llm.with_structured_output(Queries)
        chain = prompt.generate_prompt() | llm

        result = await chain.ainvoke(query.content)

        return [Query.from_str(q) for q in result.queries]


if __name__ == "__main__":
    configure_opik()
    query = Query.from_str("Give me research papers that mention the use of LLM in the cybersecurity world.")
    query_expander = QueryExpansion()
    loop = asyncio.get_event_loop()
    expanded_queries = loop.run_until_complete(query_expander.generate(query))
    for expanded_query in expanded_queries:
        logger.info(expanded_query.content)


from typing import List
from keybert import KeyBERT
import opik
from loguru import logger
from langchain_openai.chat_models import AzureChatOpenAI
import asyncio
from openai import BaseModel
from pydantic import PrivateAttr

from src.domain.prompt import Prompt
from src.domain.queries import Query
from src.tools.rag.base import RAGStep
from src.utils.constants import OPENAI_GPT4O_MINI_DEPLOYMENT_ID as GPT4_MINI, OPENAI_API_GPTO_VERSION as API_VER
from src.utils.config import config
from src.monitoring.opik import configure_opik

prompt = Prompt(
   """
    You are a expert in the cybersecurity domain. 
    *Task:* Please help the user find the most interesting academic papers to read in cybersecurity and AI world.
    You will get a list of keywords the user is interested in. Please create new queries from those keywords.
    Each new query should focus on a different of aspect or subdomain.
    
    Return a list of 3 new queries.
    
    
    --------------------------------
    *User Query Keywords:* {query}
    """
)

class Queries(BaseModel):
    queries: List[str]

class QueryKwExtraction(RAGStep):
    """
    QueryKwExtraction class that inherits from RAGStep and is responsible for extracting keywords from a query using KeyBERT.

    Attributes:
        name (str): The name of the step.
        _kw_extractor (KeyBERT): An instance of KeyBERT used for keyword extraction.
    """
    name = "query_kw_extraction"
    _kw_extractor = PrivateAttr()

    def __init__(self):
        self._kw_extractor = KeyBERT()

    @opik.track(name="QueryKwExtraction.generate")
    async def agenerate(self, query: Query) ->Query:
        """
        Asynchronously generates a new Query object with extracted keywords from the given query.

        Args:
            query (Query): The query object containing the search query.

        Returns:
            Query: A new Query object created from the extracted keywords.
        """
        kw_scores = self._kw_extractor.extract_keywords(query.content, keyphrase_ngram_range=(2, 2), stop_words='english',
                              use_mmr=True, diversity=0.3)
        kws = [k_s[0] for k_s in kw_scores]
        return Query.from_kw(kws)

class QueryExpansion:
    """
    QueryExpansion class responsible for expanding a query using a language model and keyword extraction.

    Attributes:
        name (str): The name of the step.
        _llm (AzureChatOpenAI): An instance of AzureChatOpenAI used for generating expanded queries.
        _kw_extractor (QueryKwExtraction): An instance of QueryKwExtraction used for extracting keywords from the query.
    """
    name = "query_expansion"
    _llm = PrivateAttr()
    _kw_extractor = PrivateAttr()

    def __init__(self):
        self._llm = AzureChatOpenAI(
            azure_endpoint=config.openai.url,
            openai_api_version=API_VER,
            openai_api_key=config.openai.api_key,
            deployment_name=GPT4_MINI,
            temperature=0
        )
        self._kw_extractor = QueryKwExtraction()

    @opik.track(name="QueryExpansion.generate")
    async def agenerate(self, query: Query) -> List[Query]:
        """
        Asynchronously generates a list of expanded queries based on the given query.

        Args:
            query (Query): The query object containing the search query.

        Returns:
            List[Query]: A list of expanded Query objects.
        """

        llm = self._llm.with_structured_output(Queries)
        chain = prompt.generate_prompt() | llm

        result = await chain.ainvoke(dict(query=query.keywords))
        return [Query.from_str(q) for q in result.queries]








if __name__ == "__main__":
    configure_opik()
    query = Query.from_str("Give me research papers that mention the use of LLM in the cybersecurity world.")
    query_expander = QueryExpansion()
    loop = asyncio.get_event_loop()
    expanded_queries = loop.run_until_complete(query_expander.generate(query))
    for expanded_query in expanded_queries:
        logger.info(expanded_query.content)


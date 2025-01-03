
from typing import Any
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import PrivateAttr
from loguru import logger
from opik import track

from src.domain.document import Paper, Report
from src.domain.queries import Query
from src.tools.rag.base import RAGStep
from src.domain.prompt import Prompt
from src.utils.constants import OPENAI_GPT4O_DEPLOYMENT_ID  as GPT4, OPENAI_API_GPTO_VERSION as API_VER
from src.utils.config import config

prompt = Prompt(
    """
    You are an expert on generating concise and beautiful markdown reports.
    
    Here is the user query: {query}
    Here are the academic papers that are relevant to the user query: {papers} 
    
    *Task*: Create a markdown report from all the papers highlighting the main contribution of each one and how is it
    relates to the user query.
    
    Sort them by the relevance score each paper has.
    
    Report: <The generated markdown report>
    """
)


class PapersReportGenerator(RAGStep):
    name: str = "report_generator"
    _llm: Any = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._llm = AzureChatOpenAI(
                azure_endpoint=config.openai.url,
                openai_api_version=API_VER,
                openai_api_key=config.openai.api_key,
                deployment_name=GPT4,
                temperature=0.5
        )

    @track(name="ReportGenerator.generate")
    async def agenerate(self, query: Query, papers: list[Paper], **kwargs) -> Report:
        """

        Asynchronously generates a markdown report based on the provided user query and relevant academic papers.

        This method leverages a language model to create a concise and well-structured markdown report. The report
        highlights the main contributions of each paper and explains how they relate to the user query. The papers
        are sorted by their relevance score.

        :param query: The user query that specifies the topic or question of interest.
        :type query: Query
        :param papers: A list of academic papers relevant to the user query.
        :type papers: list[Paper]
        :param kwargs: Additional keyword arguments that might be used for customization or configuration.
        :type kwargs: object
        :return: A list of academic papers sorted by relevance score, with the generated markdown report included.
        :rtype: list[Paper]
        """
        chain = prompt.generate_prompt() | self._llm | StrOutputParser()

        report = await chain.ainvoke(dict(query=query.content, papers=str(papers)))
        logger.info("Successfully generated a markdown report from the given query")
        return Report(content=report, papers=papers)





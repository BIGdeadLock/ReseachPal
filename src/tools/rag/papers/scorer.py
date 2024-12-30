
from typing import Any
from pydantic import PrivateAttr, BaseModel
from langchain_openai.chat_models import AzureChatOpenAI
import opik

from src.domain.document import Paper
from src.domain.prompt import Prompt
from src.domain.queries import Query
from src.tools.rag.base import RAGStep
from src.utils.config import config
from src.utils.constants import OPENAI_GPT4O_MINI_DEPLOYMENT_ID as GPT4, OPENAI_API_GPTO_VERSION as API_VER


prompt = Prompt(
    """
    You are a grader assessing relevance of a retrieved document to a user query. \n 
    Here is the retrieved academic paper: \n\n {paper} \n\n
    Here is the user query: {query} \n
    
    *Task:* Analyze the following pass criteria carefully. Read the document contains keyword(s) or semantic meaning related to the user query 
     and score the text based on the rubric defined below. 
    
    *Rubric:*
    How relevant is the academic paper to the user's query from a 0 to 5:
    0 - Not relevant at all
    1 - Somewhat contains shared keywords with the query but semantically different
    2 - Share a lot of keywords with the query but semantically different
    3 - Share a little of keywords with the query and is somewhat semantically related
    4 - Share a lot of keywords with the query and is somewhat semantically related
    5 - Share a lot of the keywords with the query and is very semantically related
    """,
)


class Score(BaseModel):
    score: float


class LlmAsJudge(RAGStep):
    # Private attributes for non-Pydantic fields
    _llm: Any = PrivateAttr()

    def __init__(self):
        super().__init__()
        model = AzureChatOpenAI(
                azure_endpoint=config.openai.url,
                openai_api_version=API_VER,
                openai_api_key=config.openai.api_key,
                deployment_name=GPT4,
                temperature=0
        )

        self._llm = model.with_structured_output(Score)

    @opik.track(name="LlmAsAJudge.generate")
    async def agenerate(self, query: Query, paper: Paper) -> float:
        chain = prompt.generate_prompt() | self._llm
        result = await chain.ainvoke(dict(query=query.content, paper=paper.title))
        return result.score

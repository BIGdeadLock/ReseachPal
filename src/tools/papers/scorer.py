
from pydantic import PrivateAttr, BaseModel
import re
from pydantic_ai import Agent
from src.tools.papers.base import Paper
from src.utils.config import config
from src.tools.papers.base import PaperScorer
from src.utils.constants import OPENAI_GPT4O_MINI_DEPLOYMENT_ID as GPT4, OPENAI_API_GPTO_VERSION as API_VER
from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIModel

PROMPT = """<CONTEXT>
user_query: {query}
paper: {paper}
</CONTEXT>

<USER INPUT>
Analyze the following pass criteria carefully and score the text based on the rubric defined below.

To perform this evaluation, you must:
1. Read the paper summary in the context and identify the main idea and approach.
2. Understand how is it answer the user query.
8. Assign a final score based on the scoring rubric.


Rubric:
How relevant is the academic paper to the user's query from a 0 to 5:
0 - Not relevant at all
5 - Very relevant
</USER INPUT>

"""

score_regex = re.compile(r"\d")

class Score(BaseModel):
    score: float


class LlmAsJudge(PaperScorer):
    # Private attributes for non-Pydantic fields
    _llm: Agent = PrivateAttr()

    def __init__(self):
        super().__init__()
        client = AsyncAzureOpenAI(
                azure_endpoint=config.openai.url,
                api_version=API_VER,
                api_key=config.openai.api_key,
        )

        model = OpenAIModel(GPT4, openai_client=client)
        self._llm = Agent(model, result_type=Score)
        # self._sampling_params = SamplingParams(temperature=config.judge.temperature)
        # self._llm = LLM(model=config.judge.name, tokenizer=config.judge.tokenizer, device="cpu")

    def score(self, query:str, paper: Paper) -> float:
        prompt = PROMPT.format(query=query, paper=paper)
        # outputs = self._llm.generate([prompt], sampling_params=self._sampling_params)
        # output = outputs[0]  # Since there's only one prompt, access the first output
        # generated_text = output.outputs[0].text  # Get the generated text
        # score = float(score_regex.search(generated_text).group())
        result = self._llm.run_sync(prompt)
        return result.data.score

    async def ascore(self, query:str, paper: Paper) -> float:
        prompt = PROMPT.format(query=query, paper=paper)
        # outputs = self._llm.generate([prompt], sampling_params=self._sampling_params)
        # output = outputs[0]  # Since there's only one prompt, access the first output
        # generated_text = output.outputs[0].text  # Get the generated text
        # score = float(score_regex.search(generated_text).group())
        result = await self._llm.run(prompt)
        return result.data.score
from pydantic_ai import Agent
from typing import List
from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIModel

from src.tools.papers.base import Paper, ResearchDependencies
from src.utils.constants import OPENAI_API_GPTO_VERSION as API_VER, OPENAI_GPT4O_DEPLOYMENT_ID as GPT4
from src.utils.config import config
from src.tools.papers.arxiv import aretrieve_papers


client = AsyncAzureOpenAI(
    azure_endpoint=config.openai.url,
    api_version=API_VER,
    api_key=config.openai.api_key,
)

model = OpenAIModel(GPT4, openai_client=client)
research_agent = Agent(model, result_type=List[Paper],deps_type=ResearchDependencies,
               model_settings={'temperature': config.researcher.temperature},
                system_prompt="""
                You are a scientific researcher agent tasked with finding the best and most novel research papers to a given
                query.
                To find papers you must construct a step-by-step plan to find the best list of keywords to search papers for
                the domain the user requested.
                
                Return the final list of papers
                """,
               retries=2,
               tools=[aretrieve_papers]
               )
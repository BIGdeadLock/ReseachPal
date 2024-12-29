from datetime import datetime

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.constants import TOP_K_DEFAULT, RELEVANT_SCORE_DEFAULT, ENV_FILE_PATH


class BaseServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH, extra='ignore')

class ArxivSettings(BaseModel):
    max_results: int = TOP_K_DEFAULT
    relevance_score_threshold: int = RELEVANT_SCORE_DEFAULT
    publish_year_threshold: datetime | None = None

class OpenAISettings(BaseModel):
    url: str
    api_key: str

class LLMJudgeSettings(BaseModel):
    hf_repo: str = "PatronusAI/glider-gguf.glider_Q5_K_M.gguf"
    tokenizer: str = "PatronusAI/glider"

class OpikSettings(BaseModel):
    api_key: str
    project: str

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH, env_nested_delimiter="__")
    openai: OpenAISettings
    opik: OpikSettings
    arxiv: ArxivSettings = ArxivSettings()
    llm_judge: LLMJudgeSettings = LLMJudgeSettings()




config = Settings()

# @environ.config(prefix="APP")
# class AppConfig:
#
#     @environ.config(prefix="ARXIV")
#     class Arxiv:
#         max_results = environ.var(help="The number of results to return", default=TOP_K_DEFAULT)
#         relevance_score_threshold = environ.var(help="The relevance score threshold to use", default=RELEVANT_SCORE_DEFAULT, converter=int)
#         publish_year_threshold = environ.var(help="The publish date's year threshold to use", default=0, converter=int)
#
#     @environ.config(prefix="OPENAI_AZURE")
#     class AzureOpenAI:
#         url = environ.var(help="Azure OpenAI URL")
#         api_key = environ.var(help="Azure OpenAI API key")
#
#     @environ.config(prefix="RESEARCHER")
#     class ResearchAgent:
#         temperature = environ.var(help="The temperature of the LLM", default=0, converter=float)
#         num_of_hypothesis = environ.var(help="The number of hypothesis to return for the query", default=2, converter=int)
#
#     @environ.config
#     class Judge:
#         name = environ.var(help="The name of the Judge", default="PatronusAI/glider-gguf.glider_Q5_K_M.gguf")
#         tokenizer = environ.var(help="The tokenizer of the Judge", default="PatronusAI/glider")
#         temperature = environ.var(help="The temperature of the LLM", default=0, converter=float)
#
#
#     arxiv = environ.group(Arxiv)
#     judge = environ.group(Judge)
#     openai = environ.group(AzureOpenAI)
#     researcher = environ.group(ResearchAgent)
#
# config = AppConfig.from_environ()
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


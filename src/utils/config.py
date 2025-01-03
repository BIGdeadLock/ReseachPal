from datetime import datetime

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

import src.utils.constants as consts


class PapersRetrieverSettings(BaseModel):
    max_results: int = consts.TOP_K_DEFAULT
    keep_top_k_results: int = consts.KEEP_TOP_K_DEFAULT
    relevance_score_threshold: int = consts.RELEVANT_SCORE_DEFAULT
    mmr_score_threshold: float = consts.MMR_THRESHOLD_DEFAULT
    publish_year_threshold: datetime | None = None

class EmbeddingsSettings(BaseModel):
    cross_encoder_model_id : str
    model_device: str = "cpu"
    text_embedding_model_id: str

class MongoDBSettings(BaseModel):
    host: str

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
    model_config = SettingsConfigDict(env_file=consts.ENV_FILE_PATH, env_nested_delimiter="__")
    openai: OpenAISettings
    opik: OpikSettings
    arxiv: PapersRetrieverSettings = PapersRetrieverSettings()
    llm_judge: LLMJudgeSettings = LLMJudgeSettings()
    mongo: MongoDBSettings
    embeddings: EmbeddingsSettings




config = Settings()


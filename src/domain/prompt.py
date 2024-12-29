
from langchain_core.prompts import PromptTemplate

from src.domain.utils import num_tokens_from_string
from src.domain.types import DataCategory



class Prompt:
    prompt: str
    num_tokens: int | None = None

    class Config:
        category = DataCategory.PROMPT

    def __init__(self, prompt: str, num_tokens: int | None = None):
        self.prompt = prompt
        self.num_tokens = num_tokens

    @classmethod
    def from_template(cls, prompt: str):
        num_tokens = num_tokens_from_string(prompt)
        return Prompt(prompt, num_tokens)

    def generate_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(template=self.prompt)
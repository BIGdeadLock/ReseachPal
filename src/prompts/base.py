from abc import ABC, abstractmethod

from pydantic import BaseModel


class PromptTemplateFactory(ABC, BaseModel):
    @abstractmethod
    def create_template(self, **kwargs) -> str:
        pass
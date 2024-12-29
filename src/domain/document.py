from dataclasses import dataclass
from datetime import datetime


@dataclass
class Paper:
    content: str
    title: str
    published: datetime
    url: str
    relevant_score: float

    def __repr__(self):
        return f"""
        *Title*: {self.title}
        *Published*: {self.published}
        **Summary**: {self.content}
        **URL**: {self.url}
        """
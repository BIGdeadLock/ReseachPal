from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class Paper:
    content: str
    title: str
    published: datetime
    url: str
    relevant_score: float
    embedding: list[float] = None

    def __hash__(self):
        return hash(self.title)

    def __eq__(self, other):
        return self.title == other.title

    def __repr__(self):
        return f"**Title:** {self.title}\n**Published:** {self.published}\n**Summary:** {self.content}\n**URL:** {self.url}\n**Relevant Score:** {self.relevant_score}\n"

@dataclass
class Report:
    content: str
    papers: List[Paper]

    def __repr__(self):
        papers = '\n\n'.join([str(p) for p in self.papers])
        return f"{self.content}\n\n---\n\n# Papers:\n\n{papers}\n"

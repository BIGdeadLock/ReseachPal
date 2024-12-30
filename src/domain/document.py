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

    def __hash__(self):
        return hash(self.title)

    def __eq__(self, other):
        return self.title == other.title

    def __repr__(self):
        return f"""
        Title: {self.title}
        Published: {self.published}
        Summary: {self.content}
        URL: {self.url}
        Relevant Score: {self.relevant_score}
        """

@dataclass
class Report:
    content: str
    papers: List[Paper]

    def __repr__(self):
        papers = '\n\n'.join([str(p) for p in self.papers])

        return f"""
        {self.content}
        
        ------------------------------------
        
        *Papers:*
        
        {papers}
        """.replace('\t', '')
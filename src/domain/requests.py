from pydantic import BaseModel, Field

class UserFeedbackRequest(BaseModel):
    content: str
    link: str
    platform: str
    feedback: int
    threshold: int = 3
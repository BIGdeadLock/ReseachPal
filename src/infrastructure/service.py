from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from threading import Thread

from src.domain.document import Document, DocumentMetadata

from src.infrastructure.opik_utils import configure_opik
from src.application.user_feedback.update import update

configure_opik()

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str


class FeedbackQuery(BaseModel):
    feedback: int
    platform: str
    link: str
    content: str


class QueryResponse(BaseModel):
    answer: str



@app.post("/rag", response_model=QueryResponse)
async def rag_endpoint(request: QueryRequest):
    try:
        # answer = rag(query=request.query)

        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/feedback")
async def feedback_endpoint(request: FeedbackQuery):
    try:

        doc = Document(
            content = request.content,
            user_score=request.feedback,
            metadata = DocumentMetadata(
                title = request.link.split("/")[-1],
                platform = request.platform,
                url = request.link
            )
        )
        # thread = Thread(target=update, args=(doc))
        # thread.name = f"update_feedback_{request.link}"
        # thread.start()
        update(doc)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

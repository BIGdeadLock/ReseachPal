from src.infrastructure.service import app  # noqa

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("tools.server:app", host="0.0.0.0", port=8000, reload=True)

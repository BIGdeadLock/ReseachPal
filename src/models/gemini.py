from google import genai

from src.config import settings


class Gemini:
    def __init__(self, model_id=settings.GEMINI_MODEL_ID):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_id = model_id

    def generate(self, query: str, schema=None):
        args = dict(
            model=self.model_id,
            contents=query,
        )
        if schema:
            args = args | dict(config=dict(response_mime_type="application/json", response_schema=schema))

        response = self.client.models.generate_content(**args)

        if not schema:
            return response.text

        else:
            return response.parsed

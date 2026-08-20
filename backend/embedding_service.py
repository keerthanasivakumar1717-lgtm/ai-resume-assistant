import os

from google import genai
from dotenv import load_dotenv


load_dotenv()


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to the project .env file.")
    return genai.Client(api_key=api_key)


def generate_embedding(text):
    response = _get_client().models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values

import os
from pathlib import Path

from google import genai
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().with_name(".env"))

_client = None


def _get_client():
    global _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to the project .env file.")
    if _client is None:
        _client = genai.Client(api_key=api_key)
    return _client


def _reset_client():
    global _client
    _client = None


def generate_embedding(text):
    try:
        response = _get_client().models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
    except Exception as error:
        if "client has been closed" not in str(error).lower():
            raise
        _reset_client()
        response = _get_client().models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )

    return response.embeddings[0].values

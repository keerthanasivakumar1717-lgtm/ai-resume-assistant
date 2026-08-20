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


def generate_answer(question, context):

    prompt = f"""
You are an AI Resume Assistant.

Answer the user's question using only the information
provided in the resume context.

Resume Context:
{context}

Question:
{question}

If the answer is not available in the resume context,
say that the information is not available in the resume.
"""

    try:
        response = _get_client().models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
    except Exception as error:
        if "client has been closed" not in str(error).lower():
            raise
        _reset_client()
        response = _get_client().models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

    return response.text
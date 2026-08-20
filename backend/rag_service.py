import os

from google import genai
from dotenv import load_dotenv


load_dotenv()


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to the project .env file.")
    return genai.Client(api_key=api_key)


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

    response = _get_client().models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text
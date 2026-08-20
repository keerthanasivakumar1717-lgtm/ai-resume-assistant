from fastapi import FastAPI, UploadFile, File
from PyPDF2 import PdfReader

from .chunking import chunk_text
from .embedding_service import generate_embedding
from .vector_store import search_embeddings, store_embeddings
from .rag_service import generate_answer


app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "AI Resume Assistant API is running"
    }


@app.post("/upload")
def upload_resume(file: UploadFile = File(...)):

    # Read the uploaded PDF
    reader = PdfReader(file.file)

    # Extract text from PDF
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Split text into chunks
    chunks = chunk_text(text)

    # Generate embeddings for each chunk
    embeddings = []

    for chunk in chunks:
        vector = generate_embedding(chunk)
        embeddings.append(vector)

    # Store chunks and embeddings
    store_embeddings(chunks, embeddings)

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "text": text,
        "chunks": chunks,
        "embedding_count": len(embeddings)
    }


@app.post("/search")
def search_resume(question: str):

    # Generate embedding for the question
    query_embedding = generate_embedding(question)

    # Search vector database
    results = search_embeddings(query_embedding)

    # Get relevant chunks
    relevant_chunks = results["documents"][0]

    # Combine chunks into context
    context = "\n\n".join(relevant_chunks)

    # Generate AI answer
    answer = generate_answer(
        question,
        context
    )

    return {
        "question": question,
        "results": relevant_chunks,
        "answer": answer
    }




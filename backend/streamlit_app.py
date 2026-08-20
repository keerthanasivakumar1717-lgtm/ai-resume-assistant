import os
from pathlib import Path

import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

try:
	from .chunking import chunk_text
	from .embedding_service import generate_embedding
	from .rag_service import generate_answer
	from .vector_store import search_embeddings, store_embeddings
except ImportError:
	from backend.chunking import chunk_text
	from backend.embedding_service import generate_embedding
	from backend.rag_service import generate_answer
	from backend.vector_store import search_embeddings, store_embeddings


load_dotenv(Path(__file__).resolve().with_name(".env"))

try:
	if not os.getenv("GEMINI_API_KEY") and st.secrets.get("GEMINI_API_KEY"):
		os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
	pass

st.set_page_config(
	page_title="AI Resume Assistant",
	page_icon="📄",
	layout="wide",
	initial_sidebar_state="expanded",
)

st.markdown(
	"""
	<style>
	.main-header { text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 10px; color: white; margin-bottom: 2rem; }
	.main-header h1 { margin: 0; font-size: 2.5rem; }
	.main-header p { margin: .5rem 0 0; font-size: 1.1rem; opacity: .9; }
	.section-header { color: #667eea; font-size: 1.5rem; margin: 1.5rem 0 1rem; border-bottom: 2px solid #667eea; padding-bottom: .5rem; }
	.info-box, .answer-box { padding: 1rem; border-radius: 5px; margin: 1rem 0; }
	.info-box { background: #e7f3ff; border-left: 4px solid #06c; }
	.answer-box { background: #f8f9fa; border-left: 4px solid #667eea; }
	</style>
	""",
	unsafe_allow_html=True,
)

if "resume_uploaded" not in st.session_state:
	st.session_state.resume_uploaded = False
	st.session_state.resume_filename = None
	st.session_state.chunks = []

st.markdown(
	"""
	<div class="main-header">
		<h1>📄 AI Resume Assistant</h1>
		<p>Upload your resume and ask questions using AI-powered search</p>
	</div>
	""",
	unsafe_allow_html=True,
)

st.sidebar.markdown("## 📤 Upload Resume")
if not os.getenv("GEMINI_API_KEY"):
	st.sidebar.warning("GEMINI_API_KEY is not configured. Add it to a .env file in the project root before processing a resume.")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file and st.sidebar.button("📤 Process Resume", use_container_width=True):
	with st.sidebar.status("Processing...", expanded=True) as status:
		try:
			reader = PdfReader(uploaded_file)
			text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
			if not text:
				raise ValueError("No text could be extracted from this PDF.")

			chunks = chunk_text(text)
			if not chunks:
				raise ValueError("The PDF did not produce any text chunks.")

			progress = st.progress(0)
			embeddings = []
			for index, chunk in enumerate(chunks):
				embeddings.append(generate_embedding(chunk))
				progress.progress((index + 1) / len(chunks))

			store_embeddings(chunks, embeddings)
			st.session_state.resume_uploaded = True
			st.session_state.resume_filename = uploaded_file.name
			st.session_state.chunks = chunks
			status.update(label="✅ Processing Complete", state="complete")
		except Exception as error:
			status.update(label="❌ Processing Failed", state="error")
			st.sidebar.error(str(error))

if st.session_state.resume_uploaded:
	st.sidebar.success(f"✅ {st.session_state.resume_filename}")
	st.sidebar.info(f"📊 Chunks: {len(st.session_state.chunks)}")

st.markdown('<h2 class="section-header">❓ Ask Questions</h2>', unsafe_allow_html=True)
if not st.session_state.resume_uploaded:
	st.markdown(
		'<div class="info-box"><strong>📌 Getting Started:</strong><br>Upload a PDF resume using the sidebar, process it, then ask questions about it.</div>',
		unsafe_allow_html=True,
	)
	if not os.getenv("GEMINI_API_KEY"):
		st.error("Questions are unavailable until GEMINI_API_KEY is configured.")
else:
	question = st.text_input("Enter your question:", placeholder="E.g., What are my technical skills?")
	if st.button("🔍 Search & Answer", disabled=not question, use_container_width=True):
		with st.spinner("🤖 Thinking..."):
			try:
				results = search_embeddings(generate_embedding(question), top_k=3)
				documents = results.get("documents") or []
				relevant_chunks = documents[0] if documents and documents[0] else []
				if not relevant_chunks:
					st.warning("⚠️ No relevant information found in your resume.")
				else:
					answer = generate_answer(question, "\n\n".join(relevant_chunks))
					st.markdown('<div class="answer-box"><h3>✨ Answer</h3></div>', unsafe_allow_html=True)
					st.markdown(answer)
					with st.expander("📚 View Source Context"):
						for index, chunk in enumerate(relevant_chunks, 1):
							st.markdown(f"**Chunk {index}:**")
							st.text(chunk)
			except Exception as error:
				st.error(f"❌ Error generating answer: {error}")

st.markdown('<h2 class="section-header">ℹ️ About</h2>', unsafe_allow_html=True)
st.markdown("Upload a resume, generate embeddings, search relevant sections, and ask Gemini questions grounded in the extracted content.")

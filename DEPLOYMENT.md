# Streamlit Cloud Deployment

Use these settings in Streamlit Cloud:

- **Repository:** `keerthanasivakumar1717-lgtm/ai-resume-assistant`
- **Branch:** `main`
- **Main file path:** `app.py`

Add this secret under **Settings > Secrets**:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
```

Do not use `backend/main.py` as the main file. That file is the FastAPI API entry point; `app.py` is the Streamlit user interface.
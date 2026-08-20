"""Start the FastAPI backend and Streamlit frontend together."""
import os
import subprocess
import sys


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=project_root,
        ),
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "backend/streamlit_app.py", "--server.port", "8501"],
            cwd=project_root,
        ),
    ]

    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait()

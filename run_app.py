#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"


def main():
    print("=" * 60)
    print("🚀 Starting DocuMind AI — Integrated RAG Assistant")
    print("=" * 60)

    base_dir = Path(__file__).parent.resolve()
    os.chdir(base_dir)

    python_executable = sys.executable

    print("\n1️⃣ Starting FastAPI Backend Server (http://127.0.0.1:8000)...")
    backend_cmd = [
        python_executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", "8000", "--reload"
    ]
    
    backend_process = subprocess.Popen(backend_cmd, cwd=str(base_dir))

    print("⏳ Pre-warming backend models and FAISS store (waiting 3 seconds)...")
    time.sleep(3)

    print("\n2️⃣ Starting Streamlit Frontend Dashboard (http://127.0.0.1:8501)...")
    frontend_cmd = [
        python_executable, "-m", "streamlit", "run", "frontend/streamlit_app.py",
        "--server.port=8501", "--server.address=127.0.0.1"
    ]

    frontend_process = subprocess.Popen(frontend_cmd, cwd=str(base_dir))

    print("\n" + "=" * 60)
    print("✅ DocuMind AI is now running!")
    print("👉 Streamlit UI: http://127.0.0.1:8501")
    print("👉 FastAPI Docs: http://127.0.0.1:8000/api/v1/docs")
    print("Press CTRL+C in this terminal to stop both servers.")
    print("=" * 60 + "\n")

    # Automatically open browser
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8501")

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down DocuMind AI services...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Done. Goodbye!")


if __name__ == "__main__":
    main()

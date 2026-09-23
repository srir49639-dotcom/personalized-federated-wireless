"""Single-Command Production Launcher for Personalized Federated Wireless Predictor.

Starts the FastAPI server and serves the responsive research dashboard frontend.
Usage:
    python run.py
"""

import sys
import threading
import time
import webbrowser
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def open_browser(url: str, delay: float = 3.0):
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass

def main():
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print("=" * 75)
    print("PERSONALIZED FEDERATED WIRELESS LINK QUALITY INTELLIGENCE")
    print("=" * 75)
    print(f"Target URL: {url}")
    print("Loading TensorFlow, Scikit-Learn & Federated Model Weights...")
    print("(Initial model warmup takes ~10-15 seconds. Please wait without interrupting)")
    print("Press Ctrl+C to stop the server once started.")
    print("=" * 75)

    threading.Thread(target=open_browser, args=(url, 4.0), daemon=True).start()

    uvicorn.run(
        "app.backend.main:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )

if __name__ == "__main__":
    main()


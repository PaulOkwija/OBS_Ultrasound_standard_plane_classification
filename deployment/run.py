"""
run.py
------
Launch the Afro-FetalNet inference server from any working directory.
Usage:
    python run.py
    python run.py --reload
    python run.py --host 0.0.0.0 --port 8080
"""

import sys
import os
import argparse

# Ensure the project root is on sys.path regardless of where this script is called from
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import uvicorn
from app.core.config import settings


def main():
    parser = argparse.ArgumentParser(description="Afro-FetalNet inference server")
    parser.add_argument("--host", default=settings.HOST)
    parser.add_argument("--port", type=int, default=settings.PORT)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"API docs:         http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,  # reload is incompatible with workers > 1
        app_dir=ROOT,
    )


if __name__ == "__main__":
    main()

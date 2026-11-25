"""
Simple Gemini connection test for Reading Passage Evidence Locator (PS 77).
Tests connectivity with the configured model (gemini-3.6-flash).
Sends prompt: 'Reply with exactly: GEMINI_OK'
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from backend.llm_handler import LLMHandler


def main():
    print("=" * 60)
    print("Gemini API Connectivity Test")
    print("=" * 60)

    handler = LLMHandler()
    print(f"Target Model : {handler.model_name}")

    if not handler.api_key:
        print("\n[ERROR] GEMINI_API_KEY is not set in environment or .env file.")
        print("Please set your API key in .env and retry.")
        sys.exit(1)

    print("Sending verification prompt: 'Reply with exactly: GEMINI_OK' ...")
    res = handler.test_connection()

    if res["status"] == "success":
        print(f"\n[SUCCESS] Response received: {res['response']}")
        print(f"[STATUS]  {res['message']}")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"\n[FAILED]  {res['message']}")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""RETRIEVAL PIPELINE (command line).
Run:  python ask.py "your question here"
"""
import sys

import pipeline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        raise SystemExit
    print(pipeline.ask(" ".join(sys.argv[1:])))

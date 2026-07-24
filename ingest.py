"""INJECTION PIPELINE (command line). Reads every file in data/ into the default space.
Run:  python ingest.py
"""
import glob
import os

import config
import pipeline


def main():
    files = [f for f in glob.glob(os.path.join(config.DATA_DIR, "*"))
             if f.lower().rsplit(".", 1)[-1] in ("pdf", "docx", "txt", "md")]
    if not files:
        print("No files found in the data/ folder. Drop your PDFs there and run again.")
        return
    total = 0
    for path in files:
        print(f"\nReading {os.path.basename(path)} ...")
        count, source, year = pipeline.ingest_path(path)  # default namespace
        total += count
        print(f"  stored {count} question(s) (year: {year})")
    print(f"\nDone. Stored {total} questions.")
    print('Ask:  python ask.py "your question"   |   Predict:  python predict.py')


if __name__ == "__main__":
    main()

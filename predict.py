"""PREDICTION LAYER (command line).
Ranks which topics repeat most across the stored papers.
Run:  python predict.py
"""
import pipeline


def main():
    total, rows = pipeline.predict()  # default namespace
    if total == 0:
        print("No data yet. Run `python ingest.py` first.")
        return
    print(f"\nAnalysed {total} questions across your past papers.\n")
    print(f"{'TOPIC':<40} {'TIMES ASKED':>12} {'YEARS SEEN':>12}")
    print("-" * 66)
    for r in rows:
        print(f"{r['topic'][:38]:<40} {r['count']:>7} ({r['pct']:>3}%) {r['years']:>11}")
    print("\nTopics at the top are the safest bets to study first.")


if __name__ == "__main__":
    main()

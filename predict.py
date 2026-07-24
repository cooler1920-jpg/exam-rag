"""PREDICTION LAYER (command line).
Uses Laplace's Rule of Succession + recency weighting + trend, then an LLM briefing.
Run:  python predict.py
"""
import pipeline


def main():
    total, rows = pipeline.predict()  # default namespace
    if total == 0:
        print("No data yet. Run `python ingest.py` first.")
        return
    print(f"\nAnalysed {total} questions.  (Probability = Laplace's Rule of Succession,")
    print("recent years weighted more; trend = regression slope.)\n")
    print(f"{'TOPIC':<34}{'LIKELY NEXT':>12}{'YRS SEEN':>10}{'TREND':>10}")
    print("-" * 66)
    for r in rows:
        print(f"{r['topic'][:32]:<34}{str(r['prob'])+'%':>12}{str(r['years'])+'/'+str(r['n_periods']):>10}{r['trend']:>10}")

    print("\n--- Study briefing (AI, from the numbers) ---")
    print(pipeline.predict_narrative(rows))


if __name__ == "__main__":
    main()

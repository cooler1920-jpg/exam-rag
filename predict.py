"""PREDICTION LAYER (command line).
Uses Laplace's Rule of Succession + recency weighting + trend, then an LLM briefing.
Run:  python predict.py
"""
import pipeline


def main():
    total, rows, _ = pipeline.predict()  # default namespace
    if total == 0:
        print("No data yet. Run `python ingest.py` first.")
        return
    print(f"\nAnalysed {total} questions.  (Probability = Laplace's Rule of Succession,")
    print("recent years weighted more; trend = regression slope.)\n")
    print(f"{'TOPIC':<32}{'LIKELY':>9}{'RANGE':>13}{'YRS':>7}{'TREND':>10}")
    print("-" * 71)
    for r in rows:
        rng = f"{r['lo']}-{r['hi']}%"
        print(f"{r['topic'][:30]:<32}{str(r['prob'])+'%':>9}{rng:>13}"
              f"{str(r['years'])+'/'+str(r['n_periods']):>7}{r['trend']:>10}")

    print("\n--- Study briefing (AI, from the numbers) ---")
    print(pipeline.predict_narrative(rows))


if __name__ == "__main__":
    main()

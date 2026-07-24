# 📚 Exam Question Predictor (RAG) — 100% free

Reads previous years' exam papers and predicts which topics are most likely to come.
Multimodal (Gemini reads diagrams + math), Self-RAG (re-ranks results), Pinecone (online memory).
Everything runs on free tiers — no payment needed.

## The two pipelines
- **Injection** (`ingest.py`): files → Gemini reads each page → split into questions → tag topic → Gemini embeddings → Pinecone.
- **Retrieval** (`ask.py`): your question → Gemini embed → Pinecone search → Gemini re-rank → Gemini writes a grounded answer.
- **Prediction** (`predict.py`): counts which topics repeat most across all papers.

## Setup (one time)

1. Install Python 3.10+ (you have it).
2. Open a terminal in this folder and install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Get 2 free API keys and put them in a `.env` file:
   - Copy `.env.example` to `.env`
   - Gemini → https://aistudio.google.com/app/apikey
   - Pinecone → https://app.pinecone.io
   - Paste each key into `.env`

## Use it

1. Put your past-paper PDFs into the `data/` folder.
   (Tip: name them like `physics_2019.pdf` so the year is detected.)
2. Teach the system your files:
   ```
   python ingest.py
   ```
3. Ask questions:
   ```
   python ask.py "which thermodynamics questions come up most?"
   ```
4. See predicted important topics:
   ```
   python predict.py
   ```

### Or use the web page (drag-and-drop)
```
streamlit run app.py
```

## Cost note
Everything uses free tiers. Gemini's free tier is rate-limited (a set number of
requests per minute/day), so ingesting many big papers may be slower — start with
a few papers first.

## Files
- `config.py` — all settings & API keys
- `rag_common.py` — the shared engine (clients + Claude/Gemini/Pinecone helpers)
- `ingest.py` — injection pipeline
- `ask.py` — retrieval pipeline
- `predict.py` — topic-frequency prediction
- `app.py` — optional web UI

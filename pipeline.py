"""The shared engine used by BOTH the command line and the web app.
Every function takes a `namespace` = one user's private space in Pinecone,
so different people's papers never mix. Empty namespace = the default (local) space.
"""
import os
import re
import uuid
from collections import Counter

import fitz  # PyMuPDF
import docx  # python-docx

import config
import rag_common as rag


def year_from_name(name):
    """Guess the exam year from the filename, e.g. 'physics_2019.pdf' -> '2019'."""
    m = re.search(r"(19|20)\d{2}", name)
    return m.group(0) if m else "unknown"


def pages_from_file(path):
    """Yield (page_number, png_bytes_or_None, plain_text_or_None) for each page."""
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        doc = fitz.open(path)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=150)
            yield i, pix.tobytes("png"), None
    elif ext == "docx":
        d = docx.Document(path)
        text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
        yield 1, None, text
    elif ext in ("txt", "md"):
        with open(path, encoding="utf-8", errors="ignore") as f:
            yield 1, None, f.read()


# --- INJECTION: read one file, store its questions in the user's namespace ---
def ingest_path(path, namespace=""):
    index = rag.get_index()
    source = os.path.basename(path)
    year = year_from_name(source)
    total = 0
    for page_no, png, text in pages_from_file(path):
        questions = rag.transcribe_page(png) if png else rag.transcribe_text(text)
        vectors = []
        for q in questions:
            qtext = (q.get("question_text") or "").strip()
            if not qtext:
                continue
            vid = str(uuid.uuid4())
            meta = {
                "text": qtext[:3000],
                "source": source,
                "page": page_no,
                "year": year,
                "topic": (q.get("topic", "").strip() or "unknown"),
                "marks": q.get("marks", ""),
            }
            vectors.append((vid, rag.embed(qtext), meta))
        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
            total += len(vectors)
    return total, source, year


# --- RETRIEVAL: answer a question from the user's namespace ---
def ask(query, namespace=""):
    index = rag.get_index()
    qvec = rag.embed(query, is_query=True)
    res = index.query(vector=qvec, top_k=config.TOP_K, include_metadata=True, namespace=namespace)
    matches = res.get("matches", [])
    if not matches:
        return "Nothing stored yet in this space. Add some papers first."

    candidates = [{"id": m["id"], "text": m["metadata"]["text"]} for m in matches]
    keep_ids = rag.rerank(query, candidates, config.KEEP_AFTER_RERANK)

    by_id = {m["id"]: m["metadata"] for m in matches}
    context = [
        {"text": by_id[i]["text"], "source": by_id[i]["source"], "page": by_id[i]["page"]}
        for i in keep_ids if i in by_id
    ]
    if not context:
        context = [
            {"text": m["metadata"]["text"], "source": m["metadata"]["source"], "page": m["metadata"]["page"]}
            for m in matches[:config.KEEP_AFTER_RERANK]
        ]
    return rag.answer(query, context)


# --- PREDICTION: count topic frequency from the user's namespace (works when deployed) ---
def predict(namespace=""):
    index = rag.get_index()
    probe = [0.1] * config.EMBED_DIM  # a fixed vector just to pull back all stored questions
    res = index.query(vector=probe, top_k=1000, include_metadata=True, namespace=namespace)
    counts = Counter()
    years = {}
    total = 0
    for m in res.get("matches", []):
        md = m["metadata"]
        topic = md.get("topic", "unknown")
        counts[topic] += 1
        years.setdefault(topic, set()).add(md.get("year", "unknown"))
        total += 1
    rows = []
    for topic, count in counts.most_common():
        yrs = len([y for y in years[topic] if y != "unknown"])
        rows.append({
            "topic": topic,
            "count": count,
            "pct": round(100 * count / total) if total else 0,
            "years": yrs,
        })
    return total, rows

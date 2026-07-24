"""The shared engine used by BOTH the command line and the web app.
Every function takes a `namespace` = one user's private space in Pinecone,
so different people's papers never mix. Empty namespace = the default (local) space.
"""
import math
import os
import re
import uuid
from collections import Counter, defaultdict

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


# --- PREDICTION: real probability + trend, not just counting ---
DECAY = 0.8  # recency weight: the newest year counts 1.0, each older year x0.8 (exponential smoothing)


def _slope(xs, ys):
    """Least-squares regression slope of ys over xs (the trend direction)."""
    n = len(xs)
    if n < 2:
        return 0.0
    sx, sy = sum(xs), sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    return (n * sxy - sx * sy) / denom if denom else 0.0


def predict(namespace=""):
    index = rag.get_index()
    probe = [0.1] * config.EMBED_DIM  # a fixed vector just to pull back all stored questions
    res = index.query(vector=probe, top_k=1000, include_metadata=True, namespace=namespace)
    matches = res.get("matches", [])
    total = len(matches)
    if total == 0:
        return 0, []

    # A "period" = an exam year (or, if the year is unknown, the paper itself).
    def period_of(md):
        y = str(md.get("year", "unknown"))
        return y if y != "unknown" else "paper:" + str(md.get("source", "?"))

    topic_pcount = defaultdict(lambda: defaultdict(int))  # topic -> period -> count
    topic_total = Counter()
    periods = set()
    numeric_years = set()
    for m in matches:
        md = m["metadata"]
        topic = md.get("topic", "unknown")
        p = period_of(md)
        topic_pcount[topic][p] += 1
        topic_total[topic] += 1
        periods.add(p)
        if p.isdigit():
            numeric_years.add(int(p))

    periods = sorted(periods)
    n = len(periods)                     # total number of past exams (n in the formula)
    current = max(numeric_years) if numeric_years else None

    def w(p):  # recency weight of a period
        return DECAY ** (current - int(p)) if (current is not None and p.isdigit()) else 1.0

    w_total = sum(w(p) for p in periods)

    rows = []
    for topic, pc in topic_pcount.items():
        appeared = [p for p in periods if pc.get(p, 0) > 0]
        s = len(appeared)                                  # exams this topic appeared in
        w_appeared = sum(w(p) for p in appeared)
        # Bayesian Beta-Binomial model. The Beta(a, b) posterior mean IS Laplace's
        # Rule of Succession: P(next) = (s + 1) / (n + 2), using recency-weighted evidence.
        a = w_appeared + 1
        b = (w_total - w_appeared) + 1
        prob = a / (a + b)
        # 95% credible interval (confidence band) from the Beta posterior (normal approx).
        var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        sd = math.sqrt(var)
        lo = max(0.0, prob - 1.96 * sd)
        hi = min(1.0, prob + 1.96 * sd)
        # Trend = regression slope of this topic's yearly counts.
        if len(numeric_years) >= 2:
            xs = sorted(numeric_years)
            ys = [pc.get(str(x), 0) for x in xs]
            sl = _slope(xs, ys)
        else:
            sl = 0.0
        trend = "rising" if sl > 0.15 else ("falling" if sl < -0.15 else "steady")
        rows.append({
            "topic": topic,
            "count": topic_total[topic],
            "years": s,
            "n_periods": n,
            "prob": round(100 * prob),
            "lo": round(100 * lo),
            "hi": round(100 * hi),
            "trend": trend,
        })
    rows.sort(key=lambda r: (-r["prob"], -r["count"]))
    # per-year counts for the line chart (numeric years only)
    years_sorted = sorted(numeric_years)
    series = [
        {"topic": topic, "year": y, "count": topic_pcount[topic].get(str(y), 0)}
        for topic in topic_pcount for y in years_sorted
    ]
    return total, rows, series


def predict_narrative(rows, top=6):
    """Have the LLM turn the computed numbers into a short study briefing."""
    if not rows:
        return ""
    lines = [
        f"{r['topic']}: {r['prob']}% likely next exam "
        f"(appeared in {r['years']} of {r['n_periods']} papers, trend {r['trend']})"
        for r in rows[:top]
    ]
    prompt = (
        "You are a study coach. Based ONLY on these computed statistics from a student's past "
        "exam papers, write a short (4-6 sentence) study-priority briefing: which topics to "
        "revise first and why, referring to the probability and trend. Do not invent any topic "
        "that is not listed.\n\n" + "\n".join(lines)
    )
    return rag._gen_text(prompt)


# --- CHAT: a conversation that remembers, grounded in the student's papers ---
def chat(history, namespace=""):
    """history: list of {"role": "user"|"assistant", "content": str}. Last item is the new question."""
    index = rag.get_index()
    question = history[-1]["content"]
    qvec = rag.embed(question, is_query=True)
    res = index.query(vector=qvec, top_k=config.KEEP_AFTER_RERANK + 3,
                      include_metadata=True, namespace=namespace)
    matches = res.get("matches", [])[:config.KEEP_AFTER_RERANK]
    context = "\n\n".join(
        f"[{m['metadata']['source']}, page {m['metadata']['page']}] {m['metadata']['text']}"
        for m in matches
    ) or "(no papers uploaded in this space yet)"

    convo = "\n".join(
        f"{'Student' if h['role'] == 'user' else 'Tutor'}: {h['content']}"
        for h in history[-6:]  # remember the last few turns
    )
    prompt = (
        "You are a friendly exam tutor helping a student with their own past papers. "
        "Prefer the exam-paper context below and cite it like [source, page] when you use it. "
        "If the answer is not in the papers, say so briefly, then you may give general study help. "
        "Keep replies clear and encouraging.\n\n"
        f"EXAM-PAPER CONTEXT:\n{context}\n\n"
        f"CONVERSATION SO FAR:\n{convo}\n\nReply as Tutor:"
    )
    return rag._gen_text(prompt)

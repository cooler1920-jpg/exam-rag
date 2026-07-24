"""The shared engine used by BOTH the command line and the web app.
Every function takes a `namespace` = one user's private space in Pinecone,
so different people's papers never mix. Empty namespace = the default (local) space.
"""
import math
import os
import re
import time
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


# --- STRUCTURED ANSWER: a professional, visual report (summary + findings + chart) ---
def answer_structured(query, namespace=""):
    index = rag.get_index()
    # 1) topic statistics (so topic/coverage questions get real numbers)
    total, rows, _ = predict(namespace)
    stats = "\n".join(
        f"- {r['topic']}: {r['prob']}% likely, asked {r['count']} times, trend {r['trend']}"
        for r in rows[:12]
    ) or "(no topics yet)"
    # 2) relevant questions from the papers
    qvec = rag.embed(query, is_query=True)
    res = index.query(vector=qvec, top_k=8, include_metadata=True, namespace=namespace)
    matches = res.get("matches", [])[:6]
    ctx = "\n\n".join(
        f"[{m['metadata']['source']}, page {m['metadata']['page']}] {m['metadata']['text']}"
        for m in matches
    ) or "(no papers uploaded yet)"

    prompt = (
        "You are a professional exam-analysis assistant. Answer the student's question as a concise, "
        "well-structured report using ONLY the data below. Keep everything short, simple and specific.\n"
        "Return JSON with exactly these keys:\n"
        '  "summary": a 1-2 sentence direct answer,\n'
        '  "findings": a list of {"point": short bold takeaway, "detail": 1-2 sentence explanation},\n'
        '  "focus": a list of short study actions (strings),\n'
        '  "breakdown": a list of {"label": string, "value": number 0-100} — ONLY when the question is '
        "about topic importance/coverage (use the topic likelihoods); otherwise an empty list,\n"
        '  "confidence": one of "High", "Medium", "Low",\n'
        '  "sources": a list of "[paper, page]" strings you actually used.\n\n'
        f"TOPIC STATISTICS:\n{stats}\n\nRELEVANT EXAM QUESTIONS:\n{ctx}\n\nSTUDENT QUESTION:\n{query}"
    )
    try:
        data = rag._gen_json(prompt)
    except Exception:
        data = {"summary": rag.answer(query, [
            {"text": m["metadata"]["text"], "source": m["metadata"]["source"], "page": m["metadata"]["page"]}
            for m in matches]), "findings": [], "focus": [], "breakdown": [], "confidence": "", "sources": []}
    # make sure every key exists and types are safe
    data.setdefault("summary", "")
    for k in ("findings", "focus", "breakdown", "sources"):
        if not isinstance(data.get(k), list):
            data[k] = []
    data.setdefault("confidence", "")
    return data


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


# --- BACKTEST: read a real paper and check how well the prediction matched ---
def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def extract_topics(path):
    """Read a paper and return its list of topics WITHOUT storing it (keeps the test fair)."""
    topics = []
    for page_no, png, text in pages_from_file(path):
        qs = rag.transcribe_page(png) if png else rag.transcribe_text(text)
        for q in qs:
            t = (q.get("topic", "") or "").strip()
            if t:
                topics.append(t)
    return topics


def compare(namespace, actual_topics, likely_threshold=50, match_threshold=0.80):
    """Compare the real paper's topics against our prediction for this space."""
    total, rows, _ = predict(namespace)
    if total == 0 or not actual_topics:
        return None

    pred = {r["topic"]: r["prob"] for r in rows}
    pred_topics = list(pred.keys())
    pred_vecs = [rag.embed(t) for t in pred_topics]

    counts = Counter(actual_topics)
    results = []
    for at in counts:
        av = rag.embed(at)
        best, best_sim = None, -1.0
        for pt, pv in zip(pred_topics, pred_vecs):
            sim = _cos(av, pv)
            if sim > best_sim:
                best, best_sim = pt, sim
        matched_to = best if best_sim >= match_threshold else None
        prob = pred[matched_to] if matched_to else 0
        results.append({"actual": at, "count": counts[at], "matched_to": matched_to,
                        "prob": prob, "hit": prob >= likely_threshold})

    a_total = len(results)
    hits = [r for r in results if r["hit"]]
    surprises = [r for r in results if not r["hit"]]
    recall = round(100 * len(hits) / a_total) if a_total else 0

    likely_pred = [t for t in pred_topics if pred[t] >= likely_threshold]
    matched_pred = {r["matched_to"] for r in hits if r["matched_to"]}
    precision = round(100 * len(matched_pred) / len(likely_pred)) if likely_pred else 0
    false_alarms = [{"topic": t, "prob": pred[t]} for t in likely_pred if t not in matched_pred]

    results.sort(key=lambda r: -r["prob"])
    return {"match_pct": recall, "precision_pct": precision, "actual_count": a_total,
            "results": results, "hits": hits, "surprises": surprises, "false_alarms": false_alarms}


# --- HISTORY: remember each accuracy check so reliability builds up over time ---
def _hist_ns(namespace):
    return (namespace or "default") + "__hist"


def save_backtest(namespace, result, paper=""):
    index = rag.get_index()
    vid = "bt-" + str(int(time.time() * 1000))
    meta = {
        "accuracy": result["match_pct"], "precision": result["precision_pct"],
        "actual": result["actual_count"], "paper": paper[:120],
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    index.upsert(vectors=[(vid, [0.1] * config.EMBED_DIM, meta)], namespace=_hist_ns(namespace))


def get_history(namespace):
    index = rag.get_index()
    res = index.query(vector=[0.1] * config.EMBED_DIM, top_k=100,
                      include_metadata=True, namespace=_hist_ns(namespace))
    items = [m["metadata"] for m in res.get("matches", [])]
    items.sort(key=lambda x: x.get("date", ""))
    return items


def reset_space(namespace):
    """Delete everything in a space (its questions and its accuracy history)."""
    index = rag.get_index()
    for ns in (namespace, _hist_ns(namespace)):
        try:
            index.delete(delete_all=True, namespace=ns)
        except Exception:
            pass


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
        "You are a friendly, professional exam tutor helping a student with their own past papers. "
        "Prefer the exam-paper context below and cite it like [source, page] when you use it. "
        "If the answer is not in the papers, say so briefly, then give general study help.\n"
        "FORMAT every reply cleanly: start with a one-line direct answer, then short **bold**-headed "
        "bullet points. Keep it concise and easy to scan — no long paragraphs.\n\n"
        f"EXAM-PAPER CONTEXT:\n{context}\n\n"
        f"CONVERSATION SO FAR:\n{convo}\n\nReply as Tutor:"
    )
    return rag._gen_text(prompt)


def chat_structured(history, namespace=""):
    """Same as chat() but returns a tidy structured reply for professional rendering."""
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
        for h in history[-6:]
    )
    prompt = (
        "You are a professional, friendly exam tutor. Answer the student's latest question using the "
        "exam-paper context and the conversation. Keep it short and easy to scan.\n"
        'Return JSON: {"answer": a 1-2 sentence direct reply, '
        '"points": list of {"point": short bold takeaway, "detail": 1 sentence} (only if useful, '
        'else empty), "sources": list of "[paper, page]" strings you used}.\n\n'
        f"EXAM-PAPER CONTEXT:\n{context}\n\nCONVERSATION:\n{convo}\n\nSTUDENT QUESTION:\n{question}"
    )
    try:
        data = rag._gen_json(prompt)
    except Exception:
        data = {"answer": rag._gen_text(prompt), "points": [], "sources": []}
    data.setdefault("answer", "")
    for k in ("points", "sources"):
        if not isinstance(data.get(k), list):
            data[k] = []
    return data

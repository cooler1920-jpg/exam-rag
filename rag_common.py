"""Shared building blocks: the API clients and the small functions that talk to them.
ingest.py, ask.py and predict.py all import from here.

100% Gemini (free tier) + Pinecone (free tier). No paid services."""
import json

from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

import config

config.check_keys()

# --- Clients (created once, reused everywhere) ---
gemini = genai.Client(api_key=config.GEMINI_API_KEY)
_pc = Pinecone(api_key=config.PINECONE_API_KEY)


def get_index():
    """Return the Pinecone index, creating it the first time."""
    existing = [i["name"] for i in _pc.list_indexes()]
    if config.PINECONE_INDEX not in existing:
        _pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    return _pc.Index(config.PINECONE_INDEX)


def embed(text, is_query=False):
    """Turn one piece of text into a list of numbers using Gemini."""
    task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    result = gemini.models.embed_content(
        model=config.EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=config.EMBED_DIM),
    )
    return result.embeddings[0].values


def _gen_text(contents):
    """Ask Gemini for a plain-text answer."""
    resp = gemini.models.generate_content(model=config.GEN_MODEL, contents=contents)
    return resp.text


def _gen_json(contents):
    """Ask Gemini for a JSON answer and parse it."""
    resp = gemini.models.generate_content(
        model=config.GEN_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(resp.text)


# --- Read one exam-paper page (this is the "multimodal" part: image in, questions out) ---
_TRANSCRIBE_PROMPT = (
    "This is one page of a previous-year exam question paper. "
    "Read it carefully and split it into individual questions. "
    "Return JSON in exactly this shape: "
    '{"questions": [{"question_text": "...", "topic": "...", "marks": "...", "question_number": "..."}]}. '
    "question_number = the number/label shown next to the question on the paper (e.g. '15', 'Q.15', "
    "'12(a)'), or empty string if none is visible. "
    "For each question: question_text = the full question (write mathematics as plain "
    "text/LaTeX and describe any diagram, graph, or figure in words so it is searchable); "
    "topic = a BROAD subject/chapter category in 1-3 words that MANY questions can share "
    "(e.g. 'Pharmacology', 'Anatomy', 'Thermodynamics', 'Modern History'). Do NOT invent "
    "narrow one-off sub-topics — keep it high-level so questions on the same subject get the "
    "SAME topic; marks = the marks shown, or empty string. "
    "If the page has no real questions (cover page, instructions), return an empty list."
)


def transcribe_page(png_bytes):
    """Send a page IMAGE to Gemini and get back a structured list of questions."""
    image = types.Part.from_bytes(data=png_bytes, mime_type="image/png")
    return _gen_json([image, _TRANSCRIBE_PROMPT]).get("questions", [])


def transcribe_text(text):
    """Same idea, but for a file that is already plain TEXT (docx/txt)."""
    return _gen_json(_TRANSCRIBE_PROMPT + "\n\nPAGE TEXT:\n" + text).get("questions", [])


# --- Self-RAG re-rank: grade rough matches, keep the genuinely relevant few ---
def rerank(query, candidates, keep):
    """Ask Gemini which of the rough matches actually answer the question, best first.
    candidates: list of dicts with 'id' and 'text'. Returns the ids to keep, in order."""
    listing = "\n\n".join(f"[{c['id']}] {c['text']}" for c in candidates)
    prompt = (
        f"User question:\n{query}\n\n"
        f"Below are candidate exam questions retrieved from a database. "
        f"Pick the {keep} MOST relevant ones and order them best first. "
        f"Ignore ones that are only loosely related. "
        f'Return JSON: {{"selected": ["id1", "id2", ...]}} using the ids in [brackets].\n\n{listing}'
    )
    data = _gen_json(prompt)
    return data.get("selected", [])


# --- Write the final grounded answer (prompt = instructions + context + query) ---
_ANSWER_SYSTEM = (
    "You are a study assistant. Answer the user's question using ONLY the exam questions "
    "provided as context. If the context does not contain the answer, say so honestly. "
    "Cite the source of each fact like [source, page]. Be clear and concise.\n\n"
)


def answer(query, context_blocks):
    """context_blocks: list of dicts with 'text', 'source', 'page'."""
    context = "\n\n".join(
        f"[{c['source']}, page {c['page']}] {c['text']}" for c in context_blocks
    )
    prompt = (
        _ANSWER_SYSTEM
        + f"CONTEXT (previous-year exam questions):\n{context}\n\nQUESTION:\n{query}"
    )
    return _gen_text(prompt)

"""All the knobs for the pipeline live here. Change these, not the other files."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API keys (read from your .env file) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")

# --- Models (all Gemini = free tier) ---
# Gemini reads the exam-paper images (diagrams + math), re-ranks, and writes answers.
GEN_MODEL = "gemini-2.5-flash"
# Gemini turns text into numbers (embeddings). We ask for 768 numbers per chunk.
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768

# --- Pinecone (the online vector database, free tier) ---
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "exam-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Retrieval settings ---
TOP_K = 20        # how many rough matches semantic search pulls back
KEEP_AFTER_RERANK = 5   # how many best matches survive the Self-RAG re-rank

# --- Folders ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MANIFEST = os.path.join(DATA_DIR, "manifest.jsonl")  # local record for topic-frequency prediction


def check_keys():
    """Fail early with a friendly message if a key is missing."""
    missing = [name for name, val in [
        ("GEMINI_API_KEY", GEMINI_API_KEY),
        ("PINECONE_API_KEY", PINECONE_API_KEY),
    ] if not val or val.startswith("paste-")]
    if missing:
        raise SystemExit(
            "Missing keys in your .env file: " + ", ".join(missing) +
            "\nCopy .env.example to .env and paste your real keys."
        )

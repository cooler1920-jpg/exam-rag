"""Public web app: each person types a private 'space name' and works only in it.

Run locally:  streamlit run app.py
When deployed on Streamlit Cloud, keys are read from the app's Secrets instead of .env.
"""
import os
import re
import tempfile

import streamlit as st

# When deployed, pull keys from Streamlit Secrets into the environment BEFORE
# importing our code (which reads the keys at import time). Locally this is a no-op.
for _k in ("GEMINI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX", "PINECONE_CLOUD", "PINECONE_REGION"):
    try:
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

import pipeline  # noqa: E402  (must come after secrets are loaded)

st.set_page_config(page_title="Exam Question Predictor", page_icon="📚")
st.title("📚 Exam Question Predictor")
st.caption("Upload past papers → get the topics most likely to come, with cited answers.")

# --- Each user picks a private space (their papers stay separate from everyone else's) ---
raw = st.text_input("Your space name (any word — this keeps your papers private)", value="")
namespace = re.sub(r"[^a-z0-9_-]", "", raw.strip().lower())
if not namespace:
    st.info("Type a space name above to begin (e.g. your name or a class code).")
    st.stop()
st.success(f"Working in space: **{namespace}**")

# --- 1. Add papers ---
st.header("1. Add your past papers")
uploaded = st.file_uploader(
    "Upload PDF / Word / text files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True
)
if uploaded and st.button("Learn these files"):
    prog = st.progress(0.0)
    total = 0
    for i, f in enumerate(uploaded, start=1):
        suffix = "." + f.name.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name
        # keep the real filename (so the year is detected) by renaming next to the temp file
        nice_path = os.path.join(os.path.dirname(tmp_path), f.name)
        os.replace(tmp_path, nice_path)
        with st.spinner(f"Reading {f.name} ..."):
            count, _, year = pipeline.ingest_path(nice_path, namespace=namespace)
        os.remove(nice_path)
        total += count
        st.write(f"✓ {f.name}: stored {count} question(s) (year {year})")
        prog.progress(i / len(uploaded))
    st.success(f"Done — learned {total} questions into your space.")

# --- 2. Ask ---
st.header("2. Ask a question")
q = st.text_input("e.g. 'Which thermodynamics questions come up most?'")
if st.button("Ask") and q:
    with st.spinner("Searching, re-ranking, and writing the answer..."):
        st.markdown(pipeline.ask(q, namespace=namespace))

# --- 3. Predict ---
st.header("3. Predicted important topics")
st.caption(
    "Probability uses Laplace's Rule of Succession  P(next) = (s + 1) / (n + 2)  "
    "— s = exams a topic appeared in, n = total exams — with recent years weighted more "
    "(exponential smoothing). Trend comes from a regression slope."
)
if st.button("Predict topics"):
    with st.spinner("Computing probabilities and trend..."):
        total, rows = pipeline.predict(namespace=namespace)
    if total == 0:
        st.info("No papers in this space yet — add some above first.")
    else:
        st.caption(f"Analysed {total} questions.")
        st.dataframe(
            [{"Topic": r["topic"], "Likely next exam": f"{r['prob']}%",
              "Times asked": r["count"], "Years seen": f"{r['years']}/{r['n_periods']}",
              "Trend": r["trend"]}
             for r in rows],
            use_container_width=True, hide_index=True,
        )
        with st.spinner("Writing your study briefing..."):
            st.markdown("### 🎯 Study briefing")
            st.markdown(pipeline.predict_narrative(rows))

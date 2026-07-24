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
    with st.spinner("Computing probabilities, trend and confidence..."):
        total, rows, series = pipeline.predict(namespace=namespace)
    if total == 0:
        st.info("No papers in this space yet — add some above first.")
    else:
        import pandas as pd
        import altair as alt
        df = pd.DataFrame(rows)
        top = rows[0]

        # --- Dashboard tiles ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions", total)
        c2.metric("Topics", len(rows))
        c3.metric("Past exams", top["n_periods"])
        c4.metric("Top pick", f"{top['prob']}%", top["topic"])

        # --- Bar chart: probability per topic, colored by trend, with confidence range ---
        st.markdown("#### How likely is each topic?")
        color = alt.Color("trend:N",
                          scale=alt.Scale(domain=["rising", "steady", "falling"],
                                          range=["#2f9e5f", "#9aa0ad", "#d9534f"]),
                          legend=alt.Legend(title="Trend"))
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("prob:Q", title="Likely to appear next exam (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("topic:N", sort="-x", title=None),
            color=color, tooltip=["topic", "prob", "lo", "hi", "trend", "count"],
        )
        band = alt.Chart(df).mark_rule(color="#555", size=2).encode(
            x="lo:Q", x2="hi:Q", y=alt.Y("topic:N", sort="-x"),
        )
        st.altair_chart((bars + band).properties(height=max(200, 42 * len(rows))),
                        use_container_width=True)
        st.caption("Bar = probability · thin line = 95% confidence range · green rising / red falling.")

        col_l, col_r = st.columns(2)
        # --- Line graph: each topic over the years (the real trend) ---
        with col_l:
            st.markdown("#### Trend over the years")
            if series:
                top_topics = [r["topic"] for r in rows[:6]]
                sdf = pd.DataFrame(series)
                sdf = sdf[sdf["topic"].isin(top_topics)]
                line = alt.Chart(sdf).mark_line(point=True).encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y("count:Q", title="Questions"),
                    color=alt.Color("topic:N", legend=alt.Legend(title="Topic", orient="bottom")),
                    tooltip=["topic", "year", "count"],
                ).properties(height=300)
                st.altair_chart(line, use_container_width=True)
            else:
                st.caption("Name files with the year (e.g. physics_2019.pdf) to see year-by-year trends.")
        # --- Donut: share of questions by topic ---
        with col_r:
            st.markdown("#### Share of questions")
            donut = alt.Chart(df).mark_arc(innerRadius=55).encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("topic:N", legend=alt.Legend(title="Topic", orient="bottom")),
                tooltip=["topic", "count"],
            ).properties(height=300)
            st.altair_chart(donut, use_container_width=True)

        # --- Full numbers (tucked away) ---
        with st.expander("See the full numbers"):
            st.dataframe(
                [{"Topic": r["topic"], "Likely next exam": f"{r['prob']}%",
                  "Confidence range": f"{r['lo']}–{r['hi']}%",
                  "Times asked": r["count"], "Years seen": f"{r['years']}/{r['n_periods']}",
                  "Trend": r["trend"]}
                 for r in rows],
                use_container_width=True, hide_index=True,
            )
            st.caption("Confidence range = 95% Bayesian credible interval (Beta-Binomial). "
                       "Wide range = few papers; add more for a sharper prediction.")

        with st.spinner("Writing your study briefing..."):
            st.markdown("### 🎯 Study briefing")
            st.markdown(pipeline.predict_narrative(rows))

# --- 4. Chat with your papers (remembers the conversation) ---
st.header("4. Chat with your papers")
st.caption("Have a back-and-forth conversation. Answers stay grounded in your uploaded papers.")
chat_key = f"chat_{namespace}"
st.session_state.setdefault(chat_key, [])

for msg in st.session_state[chat_key]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("Ask anything about your papers…")
if user_msg:
    st.session_state[chat_key].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = pipeline.chat(st.session_state[chat_key], namespace=namespace)
        st.markdown(reply)
    st.session_state[chat_key].append({"role": "assistant", "content": reply})

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

with st.expander("ℹ️ How to use this app"):
    st.markdown(
        "1. **Add past papers** (section 1) → click *Learn these files*.\n"
        "2. Click **Predict topics** (section 3) — likely topics, charts and a study plan.\n"
        "3. **Check accuracy** (section 4) — upload the real recent paper to score the prediction.\n"
        "4. **Chat** (section 5) — ask anything about your papers.\n\n"
        "Tip: name files with the year (e.g. `physics_2019.pdf`) for the best trends."
    )
with st.expander("🔧 Manage this space"):
    st.caption("This permanently deletes all papers, chat and history in the current space.")
    if st.button("Reset this space"):
        pipeline.reset_space(namespace)
        st.session_state.pop(f"chat_{namespace}", None)
        st.session_state.pop(f"init_{namespace}", None)
        st.success("Space cleared. Upload papers to start again.")

st.caption("🕒 Your papers, chats and history are kept for 15 days, then automatically deleted.")

# One-time per session for this space: auto-delete data older than 15 days, restore saved chat.
if not st.session_state.get(f"init_{namespace}"):
    try:
        pipeline.purge_old(namespace)
    except Exception:
        pass
    try:
        st.session_state[f"chat_{namespace}"] = pipeline.load_chat(namespace)
    except Exception:
        st.session_state[f"chat_{namespace}"] = []
    st.session_state[f"init_{namespace}"] = True

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
q = st.text_input("e.g. 'Which topics cover more than 80%? Explain what to focus on.'")
if st.button("Ask") and q:
    with st.spinner("Analysing your papers and writing a report..."):
        rep = pipeline.answer_structured(q, namespace=namespace)

    # Executive summary
    if rep.get("summary"):
        st.markdown("#### 📌 Summary")
        st.info(rep["summary"])

    # Visual breakdown (bar chart) — for topic/coverage questions
    bd = rep.get("breakdown") or []
    clean = []
    for item in bd:
        try:
            clean.append({"label": str(item.get("label", "")), "value": float(item.get("value", 0))})
        except Exception:
            pass
    if clean:
        import pandas as pd
        import altair as alt
        st.markdown("#### 📊 Breakdown")
        bdf = pd.DataFrame(clean)
        chart = alt.Chart(bdf).mark_bar(cornerRadiusEnd=3, color="#4C8BF5").encode(
            x=alt.X("value:Q", title="%", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=["label", "value"],
        ).properties(height=max(160, 38 * len(clean)))
        st.altair_chart(chart, use_container_width=True)

    # Key findings — each is a clickable expander with more detail
    findings = rep.get("findings") or []
    if findings:
        st.markdown("#### ✅ Key points  *(click to expand)*")
        for f in findings:
            point = f.get("point", "") if isinstance(f, dict) else str(f)
            detail = f.get("detail", "") if isinstance(f, dict) else ""
            with st.expander("✅  " + point):
                st.write(detail or "—")

    # What to focus on
    focus = rep.get("focus") or []
    if focus:
        st.markdown("#### 🎯 What to focus on")
        st.markdown("\n".join(f"- {x}" for x in focus))

    # Confidence + sources
    foot = []
    if rep.get("confidence"):
        foot.append(f"**Confidence:** {rep['confidence']}")
    if rep.get("sources"):
        foot.append("**Sources:** " + " · ".join(rep["sources"]))
    if foot:
        st.caption("  |  ".join(foot))

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

# --- 4. Backtest: how accurate was the prediction vs the real paper ---
st.header("4. Check accuracy — compare with the real paper")
st.caption("Upload the ACTUAL recent exam paper. We check how many of its topics our prediction "
           "got right — a real accuracy score (backtest).")
actual_file = st.file_uploader("Upload the real / recent exam paper",
                               type=["pdf", "docx", "txt", "md"], key="actual")
if actual_file and st.button("Compare with my prediction"):
    suffix = "." + actual_file.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(actual_file.getbuffer())
        tmp_path = tmp.name
    nice_path = os.path.join(os.path.dirname(tmp_path), actual_file.name)
    os.replace(tmp_path, nice_path)
    with st.spinner("Reading the real paper and comparing to your prediction..."):
        actual_topics = pipeline.extract_topics(nice_path)
        result = pipeline.compare(namespace, actual_topics)
    os.remove(nice_path)

    if result is None:
        st.info("Add past papers first (section 1), so there's a prediction to compare against.")
    elif result["actual_count"] == 0:
        st.info("Couldn't read any topics from that paper — try another file.")
    else:
        a, b, c = st.columns(3)
        a.metric("Prediction accuracy", f"{result['match_pct']}%",
                 help="Of the topics that actually appeared, how many we flagged as likely (recall).")
        b.metric("Precision", f"{result['precision_pct']}%",
                 help="Of the topics we flagged as likely, how many actually appeared.")
        c.metric("Topics in real paper", result["actual_count"])

        m, p = result["match_pct"], result["precision_pct"]
        verdict = ("Strong result — the prediction is reliable for this subject." if m >= 80
                   else "Decent — add a few more past papers to push accuracy higher." if m >= 50
                   else "The prediction needs more history — add more past papers to make it reliable.")
        st.markdown("#### 📌 Summary")
        st.info(f"The prediction correctly covered **{m}%** of the topics that actually appeared, "
                f"and **{p}%** of the topics we flagged did come. {verdict}")

        st.markdown("#### Topic by topic")
        st.dataframe(
            [{"Topic in the real paper": r["actual"],
              "We had predicted": (f"{r['prob']}%" if r["matched_to"] else "— not in past papers"),
              "Result": "✅ matched" if r["hit"] else "⚠️ we missed this"}
             for r in result["results"]],
            use_container_width=True, hide_index=True,
        )
        if result["false_alarms"]:
            st.markdown("**We flagged these as likely, but they did NOT appear:** "
                        + ", ".join(f"{f['topic']} ({f['prob']}%)" for f in result["false_alarms"]))
        st.success(f"✅ Our prediction correctly covered **{result['match_pct']}%** of the topics "
                   f"that actually came in the exam.")
        st.caption("Add more past papers to push this accuracy higher — that's how the app proves itself.")

        # Save to this space's accuracy history and show how accuracy tracks over time.
        try:
            pipeline.save_backtest(namespace, result, actual_file.name)
        except Exception:
            pass
        hist = pipeline.get_history(namespace)
        if hist:
            st.markdown("#### Accuracy history for this space")
            import pandas as pd
            import altair as alt
            hdf = pd.DataFrame(hist)
            hline = alt.Chart(hdf).mark_line(point=True).encode(
                x=alt.X("date:N", title="Checked at", sort=None),
                y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
                tooltip=["date", "accuracy", "precision", "paper"],
            ).properties(height=240)
            st.altair_chart(hline, use_container_width=True)
            avg = round(sum(h["accuracy"] for h in hist) / len(hist))
            st.caption(f"Average accuracy across {len(hist)} check(s): {avg}%.")

# --- 5. Chat with your papers (remembers the conversation) ---
st.header("5. Chat with your papers")
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
            rep = pipeline.chat_structured(st.session_state[chat_key], namespace=namespace)
        md = rep.get("answer", "")
        for p in rep.get("points", []):
            if isinstance(p, dict) and p.get("point"):
                md += f"\n- **{p['point']}** — {p.get('detail', '')}"
        st.markdown(md)
        if rep.get("sources"):
            st.caption("Sources: " + " · ".join(rep["sources"]))
    st.session_state[chat_key].append({"role": "assistant", "content": md})
    try:  # persist so the conversation survives a refresh
        pipeline.save_chat_message(namespace, "user", user_msg)
        pipeline.save_chat_message(namespace, "assistant", md)
    except Exception:
        pass

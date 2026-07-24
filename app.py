"""Public web app — a clean, chat-first Exam Question Predictor.

Run locally:  streamlit run app.py
When deployed on Streamlit Cloud, keys are read from the app's Secrets.
"""
import os
import re
import tempfile

import streamlit as st

# On Streamlit Cloud, pull keys from Secrets into the environment BEFORE importing our code.
for _k in ("GEMINI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX", "PINECONE_CLOUD", "PINECONE_REGION"):
    try:
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

import pipeline  # noqa: E402  (must come after secrets are loaded)

st.set_page_config(page_title="Exam Question Predictor", page_icon="📚")
st.title("📚 Exam Question Predictor")
st.caption("Upload past papers, then just chat — predict likely topics, get study plans, and check accuracy.")

# --- Private space ---
raw = st.text_input("Your space name (any word — keeps your papers private)", value="")
namespace = re.sub(r"[^a-z0-9_-]", "", raw.strip().lower())
if not namespace:
    st.info("Type a space name above to begin (e.g. your name or a class code).")
    st.stop()

# One-time per session: auto-delete data older than 15 days, restore saved chat.
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

chat_key = f"chat_{namespace}"
st.session_state.setdefault(chat_key, [])

st.success(f"Working in space: **{namespace}**")
st.caption("🕒 Your papers, chats and history are kept for 15 days, then automatically deleted.")


def _save_upload(f):
    """Save an uploaded file to a temp path keeping its real name (so the year is detected)."""
    suffix = "." + f.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(f.getbuffer())
        p = tmp.name
    nice = os.path.join(os.path.dirname(p), f.name)
    os.replace(p, nice)
    return nice


# --- Papers (tucked away, expanded until they add some) ---
with st.expander("📄 Add / manage your papers", expanded=True):
    uploaded = st.file_uploader("Upload PDF / Word / text files",
                                type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if uploaded and st.button("Learn these files"):
        prog = st.progress(0.0)
        total = 0
        for i, f in enumerate(uploaded, start=1):
            path = _save_upload(f)
            with st.spinner(f"Reading {f.name}…"):
                count, _, year = pipeline.ingest_path(path, namespace=namespace)
            os.remove(path)
            total += count
            st.write(f"✓ {f.name}: {count} question(s) (year {year})")
            prog.progress(i / len(uploaded))
        st.success(f"Learned {total} questions. Now ask the assistant below.")
    st.caption("Tip: name files with the year (e.g. `physics_2019.pdf`) for the best trends.")
    if st.button("🗑️ Reset this space"):
        pipeline.reset_space(namespace)
        st.session_state.pop(chat_key, None)
        st.session_state.pop(f"init_{namespace}", None)
        st.success("Space cleared. Upload papers to start again.")


# ============ MAIN: one chat assistant with suggested questions ============
st.subheader("💬 Ask your study assistant")

SUGGESTIONS = {
    "📊 Predict topics": "Predict my most important topics and what to focus on.",
    "🎯 Cover 80%+": "Which topics cover more than 80%? Explain simply in bullets.",
    "📝 Study plan": "Give me a short study plan for the most likely topics.",
    "🧠 Explain top topic": "Explain the single most likely topic in simple points.",
}
st.caption("Try one of these, or type your own below:")
cols = st.columns(len(SUGGESTIONS))
picked = None
for col, (label, qtext) in zip(cols, SUGGESTIONS.items()):
    if col.button(label, use_container_width=True):
        picked = qtext

# Show the conversation so far
for msg in st.session_state[chat_key]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

typed = st.chat_input("Ask anything about your papers…")
user_msg = picked or typed


def _is_prediction(t):
    t = t.lower()
    keys = ["predict", "topic", "cover", "focus", "likely", "important",
            "most ", "study plan", "%", "chapter", "priorit", "score"]
    return any(k in t for k in keys)


if user_msg:
    st.session_state[chat_key].append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Analysing your papers…"):
            if _is_prediction(user_msg):
                rep = pipeline.answer_structured(user_msg, namespace=namespace)
                parts = []
                if rep.get("summary"):
                    st.info(rep["summary"])
                    parts.append(rep["summary"])
                # bar chart breakdown
                clean = []
                for it in (rep.get("breakdown") or []):
                    try:
                        clean.append({"label": str(it.get("label", "")), "value": float(it.get("value", 0))})
                    except Exception:
                        pass
                if clean:
                    import pandas as pd
                    import altair as alt
                    bdf = pd.DataFrame(clean)
                    ch = alt.Chart(bdf).mark_bar(cornerRadiusEnd=3, color="#4C8BF5").encode(
                        x=alt.X("value:Q", title="%", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("label:N", sort="-x", title=None), tooltip=["label", "value"],
                    ).properties(height=max(140, 36 * len(clean)))
                    st.altair_chart(ch, use_container_width=True)
                if rep.get("findings"):
                    st.markdown("**✅ Key points**  *(click to expand)*")
                    for f in rep["findings"]:
                        pt = f.get("point", "") if isinstance(f, dict) else str(f)
                        dt = f.get("detail", "") if isinstance(f, dict) else ""
                        with st.expander("✅  " + pt):
                            st.write(dt or "—")
                        parts.append(f"- {pt}: {dt}")
                if rep.get("focus"):
                    st.markdown("**🎯 Focus on:**")
                    st.markdown("\n".join(f"- {x}" for x in rep["focus"]))
                    parts += [f"- {x}" for x in rep["focus"]]
                foot = []
                if rep.get("confidence"):
                    foot.append(f"Confidence: {rep['confidence']}")
                if rep.get("sources"):
                    foot.append("Sources: " + " · ".join(rep["sources"]))
                if foot:
                    st.caption("  |  ".join(foot))
                assistant_md = "\n".join(parts) or "(no answer)"
            else:
                rep = pipeline.chat_structured(st.session_state[chat_key], namespace=namespace)
                md = rep.get("answer", "")
                for p in rep.get("points", []):
                    if isinstance(p, dict) and p.get("point"):
                        md += f"\n- **{p['point']}** — {p.get('detail', '')}"
                st.markdown(md)
                if rep.get("sources"):
                    st.caption("Sources: " + " · ".join(rep["sources"]))
                assistant_md = md

    st.session_state[chat_key].append({"role": "assistant", "content": assistant_md})
    try:
        pipeline.save_chat_message(namespace, "user", user_msg)
        pipeline.save_chat_message(namespace, "assistant", assistant_md)
    except Exception:
        pass


# ============ Advanced tools (tucked into expanders) ============
with st.expander("📈 Full topic dashboard (charts)"):
    if st.button("Show dashboard"):
        with st.spinner("Computing probabilities and trends…"):
            total, rows, series = pipeline.predict(namespace)
        if total == 0:
            st.info("Add some papers first.")
        else:
            import pandas as pd
            import altair as alt
            df = pd.DataFrame(rows)
            top = rows[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Questions", total)
            c2.metric("Topics", len(rows))
            c3.metric("Past exams", top["n_periods"])
            c4.metric("Top pick", f"{top['prob']}%", top["topic"])
            color = alt.Color("trend:N",
                              scale=alt.Scale(domain=["rising", "steady", "falling"],
                                              range=["#2f9e5f", "#9aa0ad", "#d9534f"]),
                              legend=alt.Legend(title="Trend"))
            bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("prob:Q", title="Likely next exam (%)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("topic:N", sort="-x", title=None), color=color,
                tooltip=["topic", "prob", "lo", "hi", "trend", "count"])
            band = alt.Chart(df).mark_rule(color="#555", size=2).encode(
                x="lo:Q", x2="hi:Q", y=alt.Y("topic:N", sort="-x"))
            st.altair_chart((bars + band).properties(height=max(200, 42 * len(rows))),
                            use_container_width=True)
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Trend over the years**")
                if series:
                    tt = [r["topic"] for r in rows[:6]]
                    sdf = pd.DataFrame(series)
                    sdf = sdf[sdf["topic"].isin(tt)]
                    ln = alt.Chart(sdf).mark_line(point=True).encode(
                        x=alt.X("year:O", title="Year"), y=alt.Y("count:Q", title="Questions"),
                        color=alt.Color("topic:N", legend=alt.Legend(title="Topic", orient="bottom")),
                        tooltip=["topic", "year", "count"]).properties(height=300)
                    st.altair_chart(ln, use_container_width=True)
                else:
                    st.caption("Name files with the year to see year-by-year trends.")
            with col_r:
                st.markdown("**Share of questions**")
                dn = alt.Chart(df).mark_arc(innerRadius=55).encode(
                    theta=alt.Theta("count:Q"),
                    color=alt.Color("topic:N", legend=alt.Legend(title="Topic", orient="bottom")),
                    tooltip=["topic", "count"]).properties(height=300)
                st.altair_chart(dn, use_container_width=True)
            with st.spinner("Writing study briefing…"):
                st.markdown("**🎯 Study briefing**")
                st.markdown(pipeline.predict_narrative(rows))

with st.expander("✅ Check accuracy against a real paper"):
    st.caption("Upload the ACTUAL recent exam paper to score how well the prediction did (backtest).")
    actual_file = st.file_uploader("Upload the real / recent exam paper",
                                   type=["pdf", "docx", "txt", "md"], key="actual")
    if actual_file and st.button("Compare with my prediction"):
        path = _save_upload(actual_file)
        with st.spinner("Reading the real paper and comparing…"):
            actual_topics = pipeline.extract_topics(path)
            result = pipeline.compare(namespace, actual_topics)
        os.remove(path)
        if result is None:
            st.info("Add past papers first, so there's a prediction to compare against.")
        elif result["actual_count"] == 0:
            st.info("Couldn't read any topics from that paper — try another file.")
        else:
            a, b, c = st.columns(3)
            a.metric("Prediction accuracy", f"{result['match_pct']}%")
            b.metric("Precision", f"{result['precision_pct']}%")
            c.metric("Topics in real paper", result["actual_count"])
            m, p = result["match_pct"], result["precision_pct"]
            verdict = ("Strong result — the prediction is reliable for this subject." if m >= 80
                       else "Decent — add a few more past papers to push accuracy higher." if m >= 50
                       else "The prediction needs more history — add more past papers.")
            st.info(f"Correctly covered **{m}%** of the topics that appeared, and **{p}%** of the "
                    f"topics we flagged did come. {verdict}")
            st.dataframe(
                [{"Topic in the real paper": r["actual"],
                  "We had predicted": (f"{r['prob']}%" if r["matched_to"] else "— not in past papers"),
                  "Result": "✅ matched" if r["hit"] else "⚠️ we missed this"}
                 for r in result["results"]],
                use_container_width=True, hide_index=True)
            if result["false_alarms"]:
                st.markdown("**We flagged these but they did NOT appear:** "
                            + ", ".join(f"{f['topic']} ({f['prob']}%)" for f in result["false_alarms"]))
            try:
                pipeline.save_backtest(namespace, result, actual_file.name)
            except Exception:
                pass
            hist = pipeline.get_history(namespace)
            if hist:
                import pandas as pd
                import altair as alt
                st.markdown("**Accuracy history for this space**")
                hdf = pd.DataFrame(hist)
                hl = alt.Chart(hdf).mark_line(point=True).encode(
                    x=alt.X("date:N", title="Checked at", sort=None),
                    y=alt.Y("accuracy:Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
                    tooltip=["date", "accuracy", "precision", "paper"]).properties(height=240)
                st.altair_chart(hl, use_container_width=True)

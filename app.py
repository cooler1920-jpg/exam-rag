"""Exam Question Predictor — a clean, ChatGPT-style app.

Left sidebar = your chats, documents, quick actions and tools.
Main area = the current conversation.
"""
import os
import re
import tempfile
import time

import streamlit as st

for _k in ("GEMINI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX", "PINECONE_CLOUD",
           "PINECONE_REGION", "FAST2SMS_API_KEY", "ADMIN_MOBILE"):
    try:
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

import pipeline  # noqa: E402

st.set_page_config(page_title="Exam Question Predictor", page_icon="📚", layout="wide")

# --- ChatGPT-style sidebar polish ---
st.markdown("""
<style>
section[data-testid="stSidebar"] { background:#0d0e12; border-right:1px solid #1c1e26; }
section[data-testid="stSidebar"] .block-container { padding-top:1rem; }
/* buttons -> clean nav rows */
section[data-testid="stSidebar"] .stButton > button{
  width:100%; background:transparent; border:1px solid transparent; color:#ececf1;
  text-align:left; justify-content:flex-start; padding:.45rem .6rem; border-radius:8px;
  font-weight:400; box-shadow:none; transition:background .12s;
}
section[data-testid="stSidebar"] .stButton > button:hover{ background:rgba(255,255,255,.07); }
section[data-testid="stSidebar"] .stButton > button:focus{ box-shadow:none; }
/* text input */
section[data-testid="stSidebar"] div[data-baseweb="input"] > div{ background:#1a1c22; border-radius:10px; border-color:#2a2d36 !important; }
section[data-testid="stSidebar"] div[data-baseweb="input"] > div:focus-within{ border-color:#4C8BF5 !important; }
/* section labels (ChatGPT 'Recents' style) */
.navlabel{ font-size:.72rem; letter-spacing:.07em; text-transform:uppercase; color:#8a8f98; margin:.7rem 0 .25rem; }
/* dividers + expanders */
section[data-testid="stSidebar"] hr{ margin:.55rem 0; border-color:#20222b; }
section[data-testid="stSidebar"] details summary{ border-radius:8px; }
section[data-testid="stSidebar"] details summary:hover{ background:rgba(255,255,255,.05); }
/* hide number-input steppers (cleaner phone field) */
[data-testid="stNumberInput"] button{ display:none !important; }
</style>
""", unsafe_allow_html=True)


def _save_upload(f):
    suffix = "." + f.name.rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(f.getbuffer())
        p = tmp.name
    nice = os.path.join(os.path.dirname(p), f.name)
    os.replace(p, nice)
    return nice


QUICK = {
    "📊 Predict topics": "Predict my most important topics and what to focus on.",
    "🎯 Cover 80%+": "Which topics cover more than 80%? Explain simply in bullets.",
    "📝 Study plan": "Give me a short study plan for the most likely topics.",
    "🧠 Explain top topic": "Explain the single most likely topic in simple points.",
}

# =================== LOGIN (name + Indian mobile, no OTP) ===================
def valid_indian_mobile(s):
    d = re.sub(r"\D", "", s or "")
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    if len(d) == 11 and d.startswith("0"):
        d = d[1:]
    return d if re.fullmatch(r"[6-9]\d{9}", d) else None


if "user" not in st.session_state:
    qp = st.query_params.get("u")
    if qp:
        mob = re.sub(r"\D", "", qp)
        u = pipeline.get_user(mob)
        if u:
            st.session_state["user"] = {"mobile": mob, "name": u.get("name", "Student"), "ns": "u" + mob}

if "user" not in st.session_state:
    st.title("📚 Exam Question Predictor")
    st.write("Predict likely exam topics from past papers — with charts, study plans and accuracy checks.")
    st.subheader("Log in / Create account")

    def _clean_name():
        st.session_state["li_name"] = re.sub(r"[^A-Za-z .]", "", st.session_state.get("li_name", ""))

    def _autofill_pwd():
        # As soon as a mobile number is typed, pre-fill the password with its last 4 digits.
        v = st.session_state.get("li_mob")
        if v is not None:
            d = re.sub(r"\D", "", str(int(v)))
            if len(d) >= 4:
                st.session_state["li_pwd"] = d[-4:]

    def _clean_pwd():
        st.session_state["li_pwd"] = re.sub(r"\D", "", st.session_state.get("li_pwd", ""))[:4]

    st.text_input("Your name", key="li_name", max_chars=40, on_change=_clean_name,
                  placeholder="Letters only")
    mob_val = st.number_input("Mobile number (India)", min_value=0, max_value=9999999999,
                              value=None, step=1, format="%d", key="li_mob",
                              on_change=_autofill_pwd, placeholder="10 digits, starts 6–9")
    st.text_input("Password (4 digits)", key="li_pwd", max_chars=4, on_change=_clean_pwd)
    st.caption("💡 Your password is the **last 4 digits of your phone number** — it fills in "
               "automatically after you type your number.")

    if st.button("Log in", use_container_width=True, type="primary"):
        name = st.session_state.get("li_name", "").strip()
        raw_mob = "" if mob_val is None else str(int(mob_val))
        m = valid_indian_mobile(raw_mob)
        pwd = st.session_state.get("li_pwd", "")
        if not name:
            st.error("Please enter your name (letters only).")
        elif not m:
            st.error("Please enter a valid 10-digit Indian mobile number (starting 6–9).")
        elif pwd != m[-4:]:
            st.error("Wrong password — it's the last 4 digits of your phone number.")
        else:
            try:
                pipeline.save_user(m, name)
            except Exception:
                pass
            st.session_state["user"] = {"mobile": m, "name": name, "ns": "u" + m}
            st.query_params["u"] = m
            st.rerun()
    st.caption("By continuing, you agree we store your name and mobile to keep your account.")
    st.stop()

user = st.session_state["user"]

# =================== SIDEBAR ===================
picked = None
with st.sidebar:
    st.markdown("## 📚 Exam Predictor")
    st.caption(f"👤 **{user['name']}** · {user['mobile']}")
    if st.button("Log out", use_container_width=True):
        st.session_state.pop("user", None)
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.rerun()
    namespace = user["ns"]

    if namespace:
        # one-time init for this space this session
        if not st.session_state.get(f"init_{namespace}"):
            try:
                pipeline.purge_old(namespace)
            except Exception:
                pass
            try:
                threads = pipeline.list_threads(namespace)
            except Exception:
                threads = []
            cur = threads[0]["thread"] if threads else "main"
            st.session_state[f"cur_{namespace}"] = cur
            try:
                st.session_state[f"chat_{namespace}"] = pipeline.load_chat(namespace, cur)
            except Exception:
                st.session_state[f"chat_{namespace}"] = []
            st.session_state[f"init_{namespace}"] = True

        cur = st.session_state.get(f"cur_{namespace}", "main")
        chat_key = f"chat_{namespace}"
        st.session_state.setdefault(chat_key, [])

        st.caption("🕒 Your data auto-deletes after 15 days.")
        st.divider()

        # ---- Chats ----
        if st.button("🆕 New chat", use_container_width=True):
            cur = "t-" + str(int(time.time()))
            st.session_state[f"cur_{namespace}"] = cur
            st.session_state[chat_key] = []
        st.markdown('<div class="navlabel">💬 Your chats</div>', unsafe_allow_html=True)
        try:
            all_threads = pipeline.list_threads(namespace)
        except Exception:
            all_threads = []
        if all_threads:
            for t in all_threads[:12]:
                mark = "🟢 " if t["thread"] == cur else ""
                if st.button(mark + t["title"], key="th_" + t["thread"], use_container_width=True):
                    st.session_state[f"cur_{namespace}"] = t["thread"]
                    st.session_state[chat_key] = pipeline.load_chat(namespace, t["thread"])
                    cur = t["thread"]
        else:
            st.caption("No chats yet — ask something below.")
        st.divider()

        # ---- Documents ----
        st.markdown('<div class="navlabel">📄 Your documents</div>', unsafe_allow_html=True)
        try:
            docs = pipeline.list_documents(namespace)
        except Exception:
            docs = []
        if docs:
            for name, cnt in docs[:15]:
                short = name if len(name) <= 30 else name[:27] + "…"
                st.caption(f"• {short}  ({cnt})")
        else:
            st.caption("No papers yet.")
        with st.expander("➕ Add papers"):
            uploaded = st.file_uploader("Upload past papers", type=["pdf", "docx", "txt", "md"],
                                        accept_multiple_files=True)
            if uploaded and st.button("Learn these files", use_container_width=True):
                prog = st.progress(0.0)
                total = 0
                for i, f in enumerate(uploaded, start=1):
                    path = _save_upload(f)
                    with st.spinner(f"Reading {f.name}…"):
                        count, _, year = pipeline.ingest_path(path, namespace=namespace)
                    os.remove(path)
                    total += count
                    prog.progress(i / len(uploaded))
                st.success(f"Learned {total} questions.")
            st.caption("Tip: name files with the year, e.g. `physics_2019.pdf`.")
        st.divider()

        # ---- Suggested (quick actions + dashboard + compare) ----
        st.markdown('<div class="navlabel">⚡ Suggested</div>', unsafe_allow_html=True)
        for label, qtext in QUICK.items():
            if st.button(label, use_container_width=True):
                picked = qtext
        if st.button("📈 Topic dashboard", use_container_width=True):
            st.session_state["dash"] = True
        with st.expander("✅ Compare with a real paper"):
            actual_file = st.file_uploader("Upload the real recent paper",
                                           type=["pdf", "docx", "txt", "md"], key="actual")
            if actual_file and st.button("Compare", use_container_width=True):
                path = _save_upload(actual_file)
                with st.spinner("Reading & comparing…"):
                    topics = pipeline.extract_topics(path)
                    result = pipeline.compare(namespace, topics)
                os.remove(path)
                if result:
                    try:
                        pipeline.save_backtest(namespace, result, actual_file.name)
                    except Exception:
                        pass
                st.session_state["acc"] = result
        st.divider()

        # ---- Settings (⚙️ account controls) ----
        with st.expander("⚙️ Settings"):
            if st.button("🗑️ Reset my data", use_container_width=True):
                pipeline.reset_space(namespace)
                for k in (chat_key, f"init_{namespace}", f"cur_{namespace}"):
                    st.session_state.pop(k, None)
                st.session_state.pop("dash", None)
                st.session_state.pop("acc", None)
                st.success("Your data was cleared.")
            st.caption("⚠️ Danger zone — this cannot be undone.")
            if st.checkbox("Yes, permanently delete my account", key="del_ack"):
                if st.button("❌ Delete my account", use_container_width=True, type="primary"):
                    pipeline.delete_account(user["mobile"])
                    for k in list(st.session_state.keys()):
                        st.session_state.pop(k, None)
                    st.query_params.clear()
                    st.rerun()

        # Owner-only: customer list (set ADMIN_MOBILE in Secrets to your number)
        admin_m = re.sub(r"\D", "", os.getenv("ADMIN_MOBILE", ""))
        if admin_m and user["mobile"] == admin_m:
            st.divider()
            st.markdown('<div class="navlabel">👑 Admin</div>', unsafe_allow_html=True)
            with st.expander("Customers"):
                import datetime
                import pandas as pd
                users = pipeline.list_users()
                st.caption(f"{len(users)} signups")
                df = pd.DataFrame([{
                    "Name": u["name"], "Mobile": u["mobile"],
                    "Joined": (datetime.datetime.fromtimestamp(u["created_at"]).strftime("%Y-%m-%d")
                               if u["created_at"] else ""),
                } for u in users])
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button("⬇ Download CSV", df.to_csv(index=False),
                                   "customers.csv", "text/csv", use_container_width=True)

# =================== MAIN ===================
cur = st.session_state.get(f"cur_{namespace}", "main")
chat_key = f"chat_{namespace}"
st.session_state.setdefault(chat_key, [])

st.title("💬 Study assistant")

# --- Dashboard panel (toggled from sidebar) ---
if st.session_state.get("dash"):
    with st.container(border=True):
        c = st.columns([6, 1])
        c[0].markdown("### 📈 Topic dashboard")
        if c[1].button("✖ Hide"):
            st.session_state.pop("dash", None)
            st.rerun()
        with st.spinner("Computing…"):
            total, rows, series = pipeline.predict(namespace)
        if total == 0:
            st.info("Add some papers first (sidebar → Documents → Add papers).")
        else:
            import pandas as pd
            import altair as alt
            top = rows[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Questions", total)
            m2.metric("Topics", len(rows))
            m3.metric("Past exams", top["n_periods"])
            m4.metric("Top pick", f"{top['prob']}%", top["topic"])
            st.caption("Showing the **top 15 most-asked topics**.")
            df = pd.DataFrame(rows[:15])
            color = alt.Color("trend:N", scale=alt.Scale(domain=["rising", "steady", "falling"],
                              range=["#2f9e5f", "#9aa0ad", "#d9534f"]), legend=alt.Legend(title="Trend"))
            bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("prob:Q", title="Likely next exam (%)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("topic:N", sort="-x", title=None), color=color,
                tooltip=["topic", "prob", "lo", "hi", "trend", "count"])
            band = alt.Chart(df).mark_rule(color="#555", size=2).encode(
                x="lo:Q", x2="hi:Q", y=alt.Y("topic:N", sort="-x"))
            st.altair_chart((bars + band).properties(height=max(220, 40 * len(df))), use_container_width=True)
            cl, cr = st.columns(2)
            with cl:
                st.markdown("**Trend over the years**")
                if series:
                    tt = [r["topic"] for r in rows[:6]]
                    sdf = pd.DataFrame(series)
                    sdf = sdf[sdf["topic"].isin(tt)]
                    ln = alt.Chart(sdf).mark_line(point=True).encode(
                        x=alt.X("year:O", title="Year"), y=alt.Y("count:Q", title="Questions"),
                        color=alt.Color("topic:N", legend=alt.Legend(title="Topic", orient="bottom")),
                        tooltip=["topic", "year", "count"]).properties(height=280)
                    st.altair_chart(ln, use_container_width=True)
                else:
                    st.caption("Name files with the year to see trends.")
            with cr:
                st.markdown("**Share of questions (top 8)**")
                top8 = rows[:8]
                others = sum(r["count"] for r in rows[8:])
                pie_rows = [{"topic": r["topic"], "count": r["count"]} for r in top8]
                if others > 0:
                    pie_rows.append({"topic": "Others", "count": others})
                dn = alt.Chart(pd.DataFrame(pie_rows)).mark_arc(innerRadius=55).encode(
                    theta=alt.Theta("count:Q"),
                    color=alt.Color("topic:N", legend=alt.Legend(title="Topic", orient="bottom")),
                    tooltip=["topic", "count"]).properties(height=280)
                st.altair_chart(dn, use_container_width=True)

# --- Accuracy panel (toggled after a comparison) ---
if st.session_state.get("acc") is not None:
    result = st.session_state["acc"]
    with st.container(border=True):
        c = st.columns([6, 1])
        c[0].markdown("### ✅ Accuracy check")
        if c[1].button("✖ Hide", key="hideacc"):
            st.session_state.pop("acc", None)
            st.rerun()
        if not result or result.get("actual_count", 0) == 0:
            st.info("Couldn't read that paper, or add past papers first.")
        else:
            m, p = result["match_pct"], result["precision_pct"]
            a, b, c2 = st.columns(3)
            a.metric("Prediction accuracy", f"{m}%")
            b.metric("Precision", f"{p}%")
            c2.metric("Topics in real paper", result["actual_count"])
            verdict = ("Strong — reliable for this subject." if m >= 80
                       else "Decent — add a few more past papers." if m >= 50
                       else "Add more past papers to make it reliable.")
            st.info(f"Correctly covered **{m}%** of the topics that appeared. {verdict}")
            st.dataframe(
                [{"Topic in real paper": r["actual"],
                  "We predicted": (f"{r['prob']}%" if r["matched_to"] else "— new"),
                  "Result": "✅ matched" if r["hit"] else "⚠️ missed"}
                 for r in result["results"]],
                use_container_width=True, hide_index=True)

# --- Conversation ---
for msg in st.session_state[chat_key]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state[chat_key]:
    st.caption("👋 Add papers in the sidebar, then ask me anything — or tap a Suggested action.")

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
                clean = []
                for it in (rep.get("breakdown") or []):
                    try:
                        clean.append({"label": str(it.get("label", "")), "value": float(it.get("value", 0))})
                    except Exception:
                        pass
                clean.sort(key=lambda x: -x["value"])
                clean = clean[:8]
                if clean:
                    import pandas as pd
                    import altair as alt
                    ch = alt.Chart(pd.DataFrame(clean)).mark_bar(cornerRadiusEnd=3, color="#4C8BF5").encode(
                        x=alt.X("value:Q", title="Likely %", scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y("label:N", sort="-x", title=None), tooltip=["label", "value"],
                    ).properties(height=max(140, 34 * len(clean)))
                    st.altair_chart(ch, use_container_width=True)
                findings = [f for f in (rep.get("findings") or []) if isinstance(f, dict)]
                if findings:
                    st.markdown("**Key points:**")
                    for f in findings[:4]:
                        st.markdown(f"- **{f.get('point', '')}** — {f.get('detail', '')}")
                        parts.append(f"- {f.get('point', '')}: {f.get('detail', '')}")
                extra = findings[4:]
                focus = rep.get("focus") or []
                if extra or focus:
                    with st.expander("📖 Read more"):
                        for f in extra:
                            st.markdown(f"- **{f.get('point', '')}** — {f.get('detail', '')}")
                        if focus:
                            st.markdown("**🎯 Focus on:**")
                            st.markdown("\n".join(f"- {x}" for x in focus))
                    parts += [f"- {f.get('point', '')}: {f.get('detail', '')}" for f in extra]
                    parts += [f"- {x}" for x in focus]
                foot = []
                if rep.get("confidence"):
                    foot.append(f"Confidence: {rep['confidence']}")
                if rep.get("sources"):
                    foot.append("Sources: " + " · ".join(rep["sources"][:3]))
                if foot:
                    st.caption("  |  ".join(foot))
                assistant_md = "\n".join(parts) or "(no answer)"
            else:
                rep = pipeline.chat_structured(st.session_state[chat_key], namespace=namespace)
                md = rep.get("answer", "")
                for pt in rep.get("points", []):
                    if isinstance(pt, dict) and pt.get("point"):
                        md += f"\n- **{pt['point']}** — {pt.get('detail', '')}"
                st.markdown(md)
                if rep.get("sources"):
                    st.caption("Sources: " + " · ".join(rep["sources"]))
                assistant_md = md

    st.session_state[chat_key].append({"role": "assistant", "content": assistant_md})
    try:
        pipeline.save_chat_message(namespace, "user", user_msg, thread=cur)
        pipeline.save_chat_message(namespace, "assistant", assistant_md, thread=cur)
    except Exception:
        pass

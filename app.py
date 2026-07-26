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
/* keep the report title + Refresh button pinned to the top while scrolling */
.st-key-reportbar{
  position: sticky; top: 2.875rem; z-index: 100;
  background: var(--background-color, #0e1117);
  padding: .3rem 0 .4rem; border-bottom: 1px solid #20222b;
}
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


def render_structured_flat(rep, with_chart=True, chart_key="c"):
    """Render a structured answer WITHOUT any inner expander (safe inside a report box)."""
    import pandas as pd
    import altair as alt
    if rep.get("summary"):
        st.markdown(f"**{rep['summary']}**")
    if with_chart:
        clean = []
        for it in (rep.get("breakdown") or []):
            try:
                clean.append({"label": str(it.get("label", "")), "value": float(it.get("value", 0))})
            except Exception:
                pass
        clean.sort(key=lambda x: -x["value"])
        clean = clean[:8]
        if clean:
            df = pd.DataFrame(clean)
            # Hover shows friendly names: Topic (label) + Likely % (value)
            tip = [alt.Tooltip("label:N", title="Topic"), alt.Tooltip("value:Q", title="Likely %")]
            ctype = st.radio("Chart type", ["📊 Bar", "🥧 Pie"], horizontal=True,
                             key=f"ct_{chart_key}", label_visibility="collapsed")
            if ctype == "🥧 Pie":
                ch = alt.Chart(df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta("value:Q", title="Likely %"),
                    color=alt.Color("label:N", legend=alt.Legend(title="Topic", orient="bottom")),
                    tooltip=tip).properties(height=340)
            else:
                ch = alt.Chart(df).mark_bar(cornerRadiusEnd=3, color="#4C8BF5").encode(
                    x=alt.X("value:Q", title="Likely %", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("label:N", sort="-x", title=None), tooltip=tip,
                ).properties(height=max(140, 34 * len(clean)))
            st.altair_chart(ch, use_container_width=True)
    for f in [f for f in (rep.get("findings") or []) if isinstance(f, dict)]:
        st.markdown(f"- **{f.get('point', '')}** — {f.get('detail', '')}")
    focus = rep.get("focus") or []
    if focus:
        st.markdown("**🎯 Focus on:**")
        for x in focus:
            st.markdown(f"- {x}")
    if not rep:
        st.caption("Couldn't build this section — try Refresh.")


def render_dashboard(total, rows, series):
    """Charts for the topic dashboard (no inner expander)."""
    import pandas as pd
    import altair as alt
    top = rows[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Questions", total)
    m2.metric("Topics", len(rows))
    m3.metric("Past exams", top["n_periods"])
    m4.metric("Top pick", f"{top['prob']}%", top["topic"])
    st.caption("Top 15 most-asked topics.")
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
                color=alt.Color("topic:N", legend=alt.Legend(
                    title="Topic", orient="bottom", columns=2, labelLimit=160)),
                tooltip=["topic", "year", "count"]).properties(height=300)
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
            color=alt.Color("topic:N", legend=alt.Legend(
                title="Topic", orient="bottom", columns=2, labelLimit=160)),
            tooltip=["topic", "count"]).properties(height=300)
        st.altair_chart(dn, use_container_width=True)


def render_topic_chart(rows, chart_key="t"):
    """Bar/Pie of REAL question counts per topic. Hover = Topic · Questions · Chance %."""
    import pandas as pd
    import altair as alt
    data = [{"Topic": r["topic"], "Questions": r["count"], "Chance": r["prob"]}
            for r in rows[:10] if (r.get("topic", "") or "").lower() != "unknown"]
    if not data:
        return
    df = pd.DataFrame(data)
    tip = [alt.Tooltip("Topic:N"), alt.Tooltip("Questions:Q", title="Questions asked"),
           alt.Tooltip("Chance:Q", title="Chance next exam (%)")]
    ctype = st.radio("Chart type", ["📊 Bar", "🥧 Pie"], horizontal=True,
                     key=f"tc_{chart_key}", label_visibility="collapsed")
    if ctype == "🥧 Pie":
        ch = alt.Chart(df).mark_arc(innerRadius=55).encode(
            theta=alt.Theta("Questions:Q"),
            color=alt.Color("Topic:N", legend=alt.Legend(
                title="Topic", orient="bottom", columns=3, labelLimit=180, symbolLimit=60)),
            tooltip=tip).properties(height=360)
    else:
        ch = alt.Chart(df).mark_bar(cornerRadiusEnd=3, color="#4C8BF5").encode(
            x=alt.X("Questions:Q", title="Number of questions asked"),
            y=alt.Y("Topic:N", sort="-x", title=None), tooltip=tip,
        ).properties(height=max(140, 34 * len(df)))
    st.altair_chart(ch, use_container_width=True)
    st.caption("Bar = how many questions came from that topic. Hover for the exact number "
               "and its chance of appearing in the next exam.")


def _clean_q(text, limit=300):
    """Make one question render as a single tidy line: no stray markdown lists,
    embedded options shown inline as (1) (2) (3)…"""
    t = (text or "").replace("\r", " ")
    t = re.sub(r"\s*\n\s*", " ", t)           # newlines -> space (stops broken lists)
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\s([1-9])[\.\)]\s+", r"  ·  (\1) ", t)  # inline options: 1. x -> · (1) x
    if len(t) > limit:
        t = t[:limit - 1].rstrip() + "…"
    return t


def _q_norm(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()[:70]


def _cite(it):
    src = []
    if it.get("q_no"):
        src.append(f"Q{it['q_no']}")
    src.append(str(it.get("source", "?")))
    src.append(f"page {it.get('page', '?')}")
    if it.get("marks"):
        src.append(f"{it['marks']} marks")
    return " · ".join(src)


def topic_detail_md(detail):
    """Year-wise list of the actual questions for one topic (paper, page, q-number) + insight.
    Repeated questions (across years) are highlighted in red 🔥 = high-yield."""
    d = detail
    if not d.get("total"):
        return f"No questions found for **{d.get('topic', '')}** yet."
    # count identical/near-identical question stems across all years → a repeat = high-yield
    stem_count = {}
    for yr in d["by_year"]:
        for it in yr["items"]:
            k = _q_norm(it["text"])
            stem_count[k] = stem_count.get(k, 0) + 1
    lines = [f"**{d['topic']} — {d['total']} question(s)** across {len(d['by_year'])} paper(s)/year(s).",
             ":red-background[🔥 repeated] = came in more than one year (high-yield) · :blue-background[Q] = asked once."]
    for yr in d["by_year"]:
        lines.append(f"\n**📅 {yr['year']} — {yr['count']} question(s):**")
        for it in yr["items"][:25]:
            q = _clean_q(it["text"])
            rep = stem_count.get(_q_norm(it["text"]), 1)
            badge = f":red-background[🔥 repeated {rep}×]" if rep >= 2 else ":blue-background[Q]"
            lines.append(f"- {badge}  {q}  \n  _{_cite(it)}_")
    peak = max(d["by_year"], key=lambda y: y["count"])
    lines.append(f"\n💡 **Insight:** **{d['topic']}** appears in **{len(d['by_year'])}** of your papers; "
                 f"the most questions came in **{peak['year']}** ({peak['count']}). "
                 f"Focus on the :red-background[🔥 repeated] ones first — they're most likely to appear again.")
    return "\n".join(lines)


def coverage_md(series):
    """Deterministic: for each exam year, what % of that year's questions each topic covers."""
    from collections import defaultdict
    yt = defaultdict(int)
    ytt = defaultdict(lambda: defaultdict(int))
    for s in series or []:
        y = str(s["year"])
        yt[y] += s["count"]
        ytt[y][s["topic"]] += s["count"]
    if not any(yt.values()):
        return ("I can't split by exam because your papers aren't year-labelled. "
                "Name files with the year (e.g. `AIIMS_2023.pdf`) and re-upload.")

    def yk(y):
        return (0, -int(y)) if y.isdigit() else (1, 0)

    lines = ["**How much each topic covers in each exam (by year):**"]
    for y in sorted(yt, key=yk):
        tot = yt[y] or 1
        lines.append(f"\n**📅 {y} — {yt[y]} questions:**")
        for t, c in sorted(ytt[y].items(), key=lambda kv: -kv[1])[:12]:
            if (t or "").lower() == "unknown" or c == 0:
                continue
            lines.append(f"- {t} — **{round(100 * c / tot)}%**  ({c} Qs)")
    lines.append("\n💡 % = that topic's share of that year's questions.")
    return "\n".join(lines)


def cover_target_md(rows, total, target=80):
    """Minimum topics (most-asked first) to cover ~target% of ALL past questions — the 80/20 plan."""
    if not total or not rows:
        return "Add some papers first, then I can tell you the smallest set of topics to cover."
    target = max(10, min(100, int(target or 80)))
    picks, cum = [], 0
    for r in rows:
        if (r.get("topic", "") or "").lower() == "unknown":
            continue
        cum += r["count"]
        picks.append((r["topic"], r["count"], round(100 * cum / total)))
        if 100 * cum / total >= target:
            break
    lines = [f"**To cover ~{target}% of all questions, study these {len(picks)} topics "
             f"(most-asked first):**"]
    for i, (t, c, cp) in enumerate(picks, 1):
        lines.append(f"{i}. **{t}** — {c} Qs  _(running total {cp}%)_")
    lines.append(f"\n💡 Just these {len(picks)} topics cover about **{picks[-1][2]}%** of every past question.")
    return "\n".join(lines)


def show_coverage(series):
    """Render a PIE chart of topic coverage for each exam year, then the exact % text. Returns md."""
    import pandas as pd
    import altair as alt
    from collections import defaultdict
    yt = defaultdict(int)
    ytt = defaultdict(lambda: defaultdict(int))
    for s in series or []:
        y = str(s["year"])
        yt[y] += s["count"]
        ytt[y][s["topic"]] += s["count"]
    md = coverage_md(series)
    if not any(yt.values()):
        st.markdown(md)
        return md

    def yk(y):
        return (0, -int(y)) if y.isdigit() else (1, 0)

    st.markdown("**How much each topic covers in each exam — pie chart per year:**")
    for y in sorted(yt, key=yk):
        tot = yt[y] or 1
        items = sorted(((t, c) for t, c in ytt[y].items()
                        if c > 0 and (t or "").lower() != "unknown"), key=lambda kv: -kv[1])
        top, others = items[:8], sum(c for _, c in items[8:])
        rows = [{"Topic": t, "Questions": c, "Share": round(100 * c / tot)} for t, c in top]
        if others > 0:
            rows.append({"Topic": "Others", "Questions": others, "Share": round(100 * others / tot)})
        st.markdown(f"**📅 {y} — {yt[y]} questions**")
        ch = alt.Chart(pd.DataFrame(rows)).mark_arc(innerRadius=55).encode(
            theta=alt.Theta("Questions:Q"),
            color=alt.Color("Topic:N", legend=alt.Legend(
                title="Topic", orient="bottom", columns=3, labelLimit=160)),
            tooltip=[alt.Tooltip("Topic:N"), alt.Tooltip("Questions:Q"),
                     alt.Tooltip("Share:Q", title="Share %")]).properties(height=320)
        st.altair_chart(ch, use_container_width=True)
    st.markdown(md)
    return md


def year_detail_md(detail):
    """Subject-wise list of the actual questions for ONE year (paper, page, q-number) + insight."""
    d = detail
    if not d.get("total"):
        return f"No questions found for **{d.get('year', '')}** yet."
    lines = [f"**📅 {d['year']} — {d['total']} question(s)** across {len(d['by_topic'])} subject(s).",
             "The bigger a subject's list, the more it was tested that year."]
    for tp in d["by_topic"]:
        # highlight the most-tested subjects of the year in red
        badge = ":red-background[🔥 heavy]" if tp["count"] >= 4 else ":blue-background[·]"
        lines.append(f"\n{badge} **{tp['topic']} — {tp['count']} question(s):**")
        for it in tp["items"][:12]:
            q = _clean_q(it["text"])
            lines.append(f"- {q}  \n  _{_cite(it)}_")
        if tp["count"] > 12:
            lines.append(f"  …and **{tp['count'] - 12} more** {tp['topic']} question(s).")
    top = d["by_topic"][0]
    lines.append(f"\n💡 **Insight:** in **{d['year']}**, **{top['topic']}** had the most questions "
                 f"({top['count']}). Focus there first when revising this year's pattern.")
    return "\n".join(lines)


def show_topic_detail(detail):
    """Chart (questions per YEAR) + year-wise question list. Renders and returns the markdown."""
    import pandas as pd
    import altair as alt
    rows = [{"Year": str(y["year"]), "Questions": y["count"]}
            for y in detail.get("by_year", []) if str(y["year"]).isdigit()]
    if rows:
        df = pd.DataFrame(rows)
        ch = alt.Chart(df).mark_bar(cornerRadiusEnd=3, color="#4C8BF5").encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Questions:Q", title="Questions asked"),
            tooltip=[alt.Tooltip("Year:O"), alt.Tooltip("Questions:Q", title="Questions asked")],
        ).properties(height=240)
        st.altair_chart(ch, use_container_width=True)
    md = topic_detail_md(detail)
    st.markdown(md)
    return md


def show_year_detail(detail):
    """Chart (questions per SUBJECT) + subject-wise question list. Renders and returns the markdown."""
    import pandas as pd
    import altair as alt
    rows = [{"Subject": t["topic"], "Questions": t["count"]}
            for t in detail.get("by_topic", [])[:12] if (t.get("topic", "") or "").lower() != "unknown"]
    if rows:
        df = pd.DataFrame(rows)
        ch = alt.Chart(df).mark_bar(cornerRadiusEnd=3, color="#4C8BF5").encode(
            x=alt.X("Questions:Q", title="Questions asked"),
            y=alt.Y("Subject:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("Subject:N"), alt.Tooltip("Questions:Q", title="Questions asked")],
        ).properties(height=max(140, 34 * len(rows)))
        st.altair_chart(ch, use_container_width=True)
    md = year_detail_md(detail)
    st.markdown(md)
    return md


def topic_in_query(msg, rows):
    """If the user's message names one of the known topics, return that topic (longest match)."""
    m = (msg or "").lower()
    hits = [r["topic"] for r in rows
            if (r.get("topic", "") or "").lower() != "unknown" and r["topic"].lower() in m]
    hits.sort(key=len, reverse=True)
    return hits[0] if hits else None

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

        st.divider()

        # ======== CHATS ========
        if st.button("➕ New chat", use_container_width=True, key="newchat"):
            cur = "t-" + str(int(time.time()))
            st.session_state[f"cur_{namespace}"] = cur
            st.session_state[chat_key] = []
        st.markdown('<div class="navlabel">💬 Recent chats</div>', unsafe_allow_html=True)
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

        # ======== PAPERS (documents + upload in one box) ========
        st.markdown('<div class="navlabel">📄 Papers</div>', unsafe_allow_html=True)
        try:
            docs = pipeline.list_documents(namespace)
        except Exception:
            docs = []
        with st.expander(f"Your papers ({len(docs)})", expanded=False):
            if docs:
                for name, cnt in docs[:15]:
                    short = name if len(name) <= 30 else name[:27] + "…"
                    st.caption(f"• {short}  ({cnt})")
            else:
                st.caption("No papers yet.")
            st.markdown("**➕ Add more papers**")
            uploaded = st.file_uploader("Upload past papers", type=["pdf", "docx", "txt", "md"],
                                        accept_multiple_files=True, label_visibility="collapsed")
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
                st.session_state.pop(f"report_{namespace}", None)  # rebuild report with new papers
                st.success(f"Learned {total} questions. Your report will refresh.")
            st.caption("Tip: name files with the year, e.g. `physics_2019.pdf`.")

        # ======== TOOLS ========
        st.markdown('<div class="navlabel">⚙️ Tools</div>', unsafe_allow_html=True)
        with st.expander("Settings"):
            st.markdown("**✅ Compare with a real paper**")
            st.caption("Upload a recent paper → we check what we predicted vs what actually came.")
            actual_file = st.file_uploader("Upload a recent real paper to check accuracy",
                                           type=["pdf", "docx", "txt", "md"], key="actual")
            if actual_file and st.button("Compare", use_container_width=True):
                path = _save_upload(actual_file)
                result = None
                try:
                    with st.spinner("Reading the paper & comparing… (a big PDF can take a minute)"):
                        topics = pipeline.extract_topics(path)
                        result = pipeline.compare(namespace, topics)
                except Exception as e:
                    result = {"error": str(e)[:200]}
                finally:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                if result and result.get("actual_count"):
                    try:
                        pipeline.save_backtest(namespace, result, actual_file.name)
                    except Exception:
                        pass
                # always store SOMETHING so the result panel shows a message (never silent)
                st.session_state["acc"] = result or {"actual_count": 0}
                st.rerun()
            st.divider()
            if st.button("🗑️ Reset my data", use_container_width=True):
                pipeline.reset_space(namespace)
                for k in (chat_key, f"init_{namespace}", f"cur_{namespace}", f"report_{namespace}"):
                    st.session_state.pop(k, None)
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

        # ======== ADMIN (owner only) ========
        admin_m = re.sub(r"\D", "", os.getenv("ADMIN_MOBILE", ""))
        if admin_m and user["mobile"] == admin_m:
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

        # ======== ACCOUNT (bottom) ========
        st.divider()
        st.caption(f"👤 **{user['name']}** · {user['mobile']}")
        st.caption("🕒 Your data auto-deletes after 15 days.")
        if st.button("Log out", use_container_width=True):
            st.session_state.pop("user", None)
            try:
                st.query_params.clear()
            except Exception:
                pass
            st.rerun()

# =================== MAIN ===================
cur = st.session_state.get(f"cur_{namespace}", "main")
chat_key = f"chat_{namespace}"
st.session_state.setdefault(chat_key, [])

st.title("💬 Study assistant")

# --- Prediction accuracy (shows at the TOP the moment you Compare a real paper) ---
if st.session_state.get("acc") is not None:
    result = st.session_state["acc"]
    with st.container(border=True):
        c = st.columns([6, 1])
        c[0].markdown("### 🎯 Prediction vs reality")
        if c[1].button("✖ Hide", key="hideacc"):
            st.session_state.pop("acc", None)
            st.rerun()
        if result.get("error"):
            st.error(f"Couldn't read that paper: {result['error']}")
        elif not result.get("actual_count"):
            st.warning("I couldn't find any questions in that file. Make sure it's a real question "
                       "paper (PDF/DOCX/text) with readable text, then try Compare again.")
        else:
            m, p = result["match_pct"], result["precision_pct"]
            k1, k2, k3 = st.columns(3)
            k1.metric("We predicted right", f"{m}%",
                      help="Of the topics that actually appeared, how many we had flagged as likely")
            k2.metric("Precision", f"{p}%",
                      help="Of our 'likely' picks, how many actually appeared")
            k3.metric("Topics in the real paper", result["actual_count"])
            verdict = ("🟢 Strong — our prediction was reliable." if m >= 80
                       else "🟡 Decent — add a few more past papers to improve." if m >= 50
                       else "🔴 Add more past papers to make predictions reliable.")
            st.info(f"We correctly predicted **{m}%** of the topics that actually appeared. {verdict}")
            hits = result.get("hits", [])
            surprises = result.get("surprises", [])
            fa = result.get("false_alarms", [])
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**✅ We predicted these — and they came:**")
                st.markdown("\n".join(f"- {h['actual']} _(predicted {h['prob']}%)_"
                                      for h in hits[:12]) or "- (none)")
            with cB:
                st.markdown("**⚠️ Came in the exam — we missed:**")
                st.markdown("\n".join(f"- {s['actual']}" for s in surprises[:12]) or "- (none)")
            if fa:
                with st.expander("🔮 We predicted these, but they didn't appear this time"):
                    st.markdown("\n".join(f"- {f['topic']} _({f['prob']}%)_" for f in fa[:15]))
            with st.expander("📄 Full topic-by-topic table"):
                st.dataframe(
                    [{"Topic in real paper": r["actual"],
                      "We predicted": (f"{r['prob']}%" if r["matched_to"] else "— not predicted"),
                      "Result": "✅ predicted" if r["hit"] else "⚠️ missed"}
                     for r in result["results"]],
                    use_container_width=True, hide_index=True)

# --- Auto report: 5 sections built automatically from the papers (cached per session) ---
report_key = f"report_{namespace}"
with st.container(key="reportbar"):  # sticky bar: stays at top while you scroll
    rc = st.columns([6, 1])
    rc[0].markdown("### 📋 Your exam report")
    if rc[1].button("🔄 Refresh"):
        st.session_state.pop(report_key, None)

if report_key not in st.session_state:
    with st.spinner("Reading your papers and preparing your report…"):
        total, rows, series = pipeline.predict(namespace)
        R = {"total": total, "rows": rows, "series": series}
        if total > 0:
            for k, q in (("topics", QUICK["📊 Predict topics"]),
                         ("cover", QUICK["🎯 Cover 80%+"]),
                         ("plan", QUICK["📝 Study plan"]),
                         ("explain", QUICK["🧠 Explain top topic"])):
                try:
                    R[k] = pipeline.answer_structured(q, namespace=namespace)
                except Exception:
                    R[k] = {}
        st.session_state[report_key] = R

R = st.session_state[report_key]
if R["total"] == 0:
    st.info("👋 Add your past papers in the sidebar (📄 Papers → Your papers → Add more papers). "
            "Your full report will appear here automatically.")
else:
    with st.expander("🎯 Most important topics", expanded=True):
        render_structured_flat(R.get("topics", {}), with_chart=False)
        render_topic_chart(R["rows"], "topics")
        names = [r["topic"] for r in R["rows"][:25]
                 if (r.get("topic", "") or "").lower() != "unknown"]
        pick = st.selectbox("🔍 See every question from a topic (year-wise)",
                            ["—"] + names, key="topicpick")
        if pick and pick != "—":
            with st.spinner("Pulling year-wise questions…"):
                detail = pipeline.topic_questions(namespace, pick)
            show_topic_detail(detail)

        # Year filter: pick a year → subject-wise question breakdown for that year
        years = sorted({str(s["year"]) for s in R.get("series", [])}, reverse=True)
        if years:
            ypick = st.selectbox("📅 Filter by year (see that year's questions, subject-wise)",
                                 ["—"] + years, key="yearpick")
            if ypick and ypick != "—":
                with st.spinner(f"Loading {ypick} questions…"):
                    ydetail = pipeline.year_questions(namespace, ypick)
                show_year_detail(ydetail)
    with st.expander("📊 Topic dashboard (charts)"):
        render_dashboard(R["total"], R["rows"], R["series"])
    with st.expander("✅ Topics that cover 80%+"):
        render_structured_flat(R.get("cover", {}), with_chart=False)
    with st.expander("📝 Study plan"):
        render_structured_flat(R.get("plan", {}), with_chart=False)
    with st.expander("🧠 Top topic explained"):
        render_structured_flat(R.get("explain", {}), with_chart=False)
    st.caption("💬 Need anything more? Just ask below.")

# --- Conversation ---
for msg in st.session_state[chat_key]:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🙋"):
            st.markdown("**❓ Question**")
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown("**✅ Answer**")
            st.markdown(msg["content"])

if not st.session_state[chat_key]:
    st.caption("👆 Your report is above. Ask me anything else about your papers below.")

typed = st.chat_input("Ask anything about your papers…")
user_msg = picked or typed


if user_msg:
    st.session_state[chat_key].append({"role": "user", "content": user_msg})
    with st.chat_message("user", avatar="🙋"):
        st.markdown("**❓ Question**")
        st.markdown(user_msg)

    with st.chat_message("assistant", avatar="🧠"):
        st.markdown("**✅ Answer**")
        with st.spinner("Analysing your papers…"):
            _R = st.session_state.get(report_key, {})
            _rows = _R.get("rows") or []
            _series = _R.get("series") or []
            _topics = [r["topic"] for r in _rows if (r.get("topic", "") or "").lower() != "unknown"]
            _years = list({str(s["year"]) for s in _series})
            # AI router: understands the question (typos/Hinglish included) and picks how to answer.
            intent = pipeline.classify_intent(user_msg, _topics, _years)
            it = intent.get("intent")
            if it == "coverage_by_year":
                assistant_md = show_coverage(_series)
            elif it == "coverage_target":
                assistant_md = cover_target_md(_rows, _R.get("total", 0), intent.get("percent") or 80)
                st.markdown(assistant_md)
            elif it == "year_questions" and intent.get("year"):
                assistant_md = show_year_detail(pipeline.year_questions(namespace, intent["year"]))
            elif it == "topic_questions" and intent.get("topic"):
                assistant_md = show_topic_detail(pipeline.topic_questions(namespace, intent["topic"]))
            else:
                # Flexible answer that obeys the student's exact wording (count, length, format).
                assistant_md = pipeline.chat_reply(st.session_state[chat_key], namespace=namespace)
                st.markdown(assistant_md)

    st.session_state[chat_key].append({"role": "assistant", "content": assistant_md})
    try:
        pipeline.save_chat_message(namespace, "user", user_msg, thread=cur)
        pipeline.save_chat_message(namespace, "assistant", assistant_md, thread=cur)
    except Exception:
        pass

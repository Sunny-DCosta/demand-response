"""OnPoint demand-response assistant — chat over your customer fleet.

Ask about one customer (name an Exp_#### id), the whole fleet, a slice, a ranking,
or score a hypothetical new customer. Every number is computed in Python; the local
LLM only routes the question and narrates the result, so the app stays useful (tables
+ charts) even when Ollama is offline.

Run:  streamlit run master/app/dashboard.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # for config
import pandas as pd
import streamlit as st
import agent, services, llm
import config as C

_ICON = str(C.LOGO) if C.LOGO.exists() else "⚡"
st.set_page_config(page_title="OnPoint DR Assistant", page_icon=_ICON, layout="wide")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1100px; }}
  /* hide the empty sidebar header strip that pushes the logo down (stable testid) */
  section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{ display: none; }}
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: 0.75rem; }}
  /* sidebar example queries as quiet, left-aligned chips */
  section[data-testid="stSidebar"] .stButton button {{
      text-align: left; justify-content: flex-start; font-weight: 400;
      font-size: 0.83rem; line-height: 1.25; padding: 0.34rem 0.6rem;
      border: 1px solid #E3EAF6; background: #FFFFFF; color: #2A3650;
  }}
  section[data-testid="stSidebar"] .stButton button:hover {{
      border-color: {C.BRAND}; color: {C.BRAND}; background: #F4F8FE;
  }}
  h1 {{ color: {C.BRAND}; }}
</style>
""", unsafe_allow_html=True)

EXAMPLES = [
    "What's in this dataset?",
    "Which region has the highest non-responder share?",
    "Top 5 regions by reliable share",
    "How many users in Oslo are reliable?",
    "Average convertibility of occasional users in Bergen",
    "Reliable share by region and signal",
    "Tell me about Exp_3973",
]

_AV = str(C.BITSY) if C.BITSY.exists() else None      # shared chat avatar (user + assistant)


def _fmt(v):
    return "—" if v is None else v


def answer_chart(d):
    """A single relevant chart shown inline with the answer (kept deliberately minimal).
    Skipped when it would be a single bar (e.g. a 'which region has the most…' top-1)."""
    if not isinstance(d, dict):
        return
    res = d.get("results")
    if res and len(res) >= 2:                                    # ranking with 2+ groups
        st.bar_chart(pd.DataFrame(res).set_index("group")["value"], height=240)
    elif d.get("tier_probabilities"):                            # prospect score
        st.bar_chart(pd.Series(d["tier_probabilities"], name="probability"), height=240)
    elif d.get("queryable_columns") and d.get("tier_sizes"):     # dataset overview
        st.bar_chart(pd.Series(d["tier_sizes"], name="customers"), height=240)


def render_result(d):
    """Render the ground-truth dict as charts/metrics/tables (not raw JSON)."""
    if not isinstance(d, dict):
        st.json(d); return

    # ---- new-customer (prospect) score ----
    if "predicted_tier" in d:
        c1, c2 = st.columns(2)
        c1.metric("Predicted tier", str(d.get("predicted_tier")).title())
        c2.metric("P(reliable)", _fmt(d.get("p_reliable")))
        tp = d.get("tier_probabilities")
        if tp:
            st.bar_chart(pd.Series(tp, name="probability"))
        st.caption(d.get("targeting", ""))
        with st.expander("inputs & detail"):
            st.json(d)
        return

    # ---- grouped / ranked result ----
    if "results" in d and d.get("results"):
        df = pd.DataFrame(d["results"])
        m = d.get("metric", {})
        col = m.get("col")
        label = (col[3:].replace("_", "-") + " share") if col and col.startswith("is_") \
            else (col or m.get("func", "value"))
        st.caption(f"Grouped by **{d.get('group_by')}** · **{label}**")
        if d.get("scope_note"):
            st.caption(f"⚠ {d['scope_note']}")
        if len(df) >= 2:                       # a single bar conveys nothing — show the table only
            st.bar_chart(df.set_index("group")["value"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        with st.expander("spec / filters applied"):
            st.json({k: v for k, v in d.items() if k != "results"})
        return

    # ---- single user ----
    if "ID" in d and "tier_raw" in d:
        c = st.columns(4)
        c[0].metric("Tier", str(d.get("tier_raw")).title())
        c[1].metric("Flag rate", _fmt(d.get("flag_rate")))
        c[2].metric("P(reliable)", _fmt(d.get("p_reliable")))
        if d.get("tier_raw") == "occasional":          # convertibility is actionable only here
            c[3].metric("Convertibility", _fmt(d.get("convertibility")))
        else:
            c[3].metric("Events", _fmt(d.get("n_events")))
        with st.expander("all facts"):
            st.json(d)
        return

    # ---- dataset overview (describe) ----
    if "queryable_columns" in d:
        c = st.columns(3)
        c[0].metric("Customers", f"{_fmt(d.get('total_users')):,}" if isinstance(d.get("total_users"), int) else _fmt(d.get("total_users")))
        c[1].metric("Regions", len(d.get("regions", [])))
        c[2].metric("Signals", len(d.get("signals", [])))
        ts = d.get("tier_sizes")
        if ts:
            st.bar_chart(pd.Series(ts, name="users"))
        qc = d["queryable_columns"]
        with st.expander("columns you can ask about"):
            st.write("**Numeric (filter / average / rank):** " + ", ".join(qc.get("numeric", [])))
            st.write("**Categorical (group / filter):** " + ", ".join(qc.get("categorical", [])))
        return

    # ---- filtered slice summary ----
    if "n_users" in d:
        if d.get("scope_note"):
            st.caption(f"⚠ {d['scope_note']}")
        c = st.columns(4)
        c[0].metric("Users", f"{d['n_users']:,}")
        c[1].metric("% of total", _fmt(d.get("pct_of_total")))
        c[2].metric("% reliable", _fmt(d.get("pct_reliable")))
        c[3].metric("% occasional", _fmt(d.get("pct_occasional")))
        means = d.get("means")
        if means:
            st.dataframe(pd.DataFrame([means]), use_container_width=True, hide_index=True)
        with st.expander("filters applied / detail"):
            st.json(d)
        return

    st.json(d)


# ── sidebar: status, KPIs, examples, schema ──────────────────────────────────
with st.sidebar:
    if C.LOGO.exists():
        st.image(str(C.LOGO), width=230)               # slightly smaller than full sidebar width
    else:
        st.markdown("## ⚡ OnPoint")
    st.caption("Demand-Response Intelligence")
    up = llm.available()
    st.markdown(f"**Model:** `{C.LLM_MODEL}`")
    st.caption(f"build {llm.BUILD}")               # changes only after a real server restart
    if not up:
        st.caption("Ollama offline — numbers still work, no prose. Start it: `ollama serve`")

    try:
        s = services.cohort_summary()
        ts = s["tier_sizes"]["raw"]
        st.markdown(
            f"**Customers:** {s.get('total_users', 0):,}  \n"
            f"**Reliable:** {ts.get('reliable', 0):,}  \n"
            f"**Occasional:** {ts.get('occasional', 0):,}  \n"
            f"**Non-responder:** {ts.get('non-responder', 0):,}"
        )
        st.markdown("**What the tiers mean** — how dependably a customer lowers "
                    "electricity use when the utility calls a demand-response event:")
        st.markdown(
            "- **Reliable** — responds to *most* events\n"
            "- **Occasional** — responds *sometimes*, not dependably\n"
            "- **Non-responder** — *rarely or never* responds"
        )
    except Exception:
        st.caption("Run the pipeline to populate KPIs.")

    st.divider()
    st.caption("Try one:")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, use_container_width=True, key=f"ex{i}"):
            st.session_state.pending = ex

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.msgs = []

    with st.expander("What can I ask?"):
        sc = services.query_schema()
        st.caption("**Filter / aggregate columns**")
        st.caption(", ".join(sc["numeric_columns"]))
        st.caption("**Group / category columns**")
        st.caption(", ".join(sc["categorical_columns"].keys()))


# ── main: chat ───────────────────────────────────────────────────────────────
st.title("Demand-Response Assistant")
st.caption("Ask about any customer or the whole fleet. Numbers are computed in Python; "
           "the LLM only routes and explains. The *computed result* panel is for debugging.")

if "msgs" not in st.session_state:
    st.session_state.msgs = []

for m in st.session_state.msgs:
    with st.chat_message(m["role"], avatar=_AV):
        st.markdown(m["content"])
        if m.get("data") is not None:
            answer_chart(m["data"])
            with st.expander("computed result (ground truth — for debugging)"):
                render_result(m["data"])

prompt = st.chat_input("Ask about your customers…") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=_AV):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=_AV):
        with st.spinner("thinking…"):
            prose, data = agent.answer(prompt)
        text = prose or "_LLM offline — showing the computed result below._"
        st.markdown(text)
        if data is not None:
            answer_chart(data)
            with st.expander("computed result (ground truth — for debugging)", expanded=not prose):
                render_result(data)
    st.session_state.msgs.append({"role": "assistant", "content": text, "data": data})

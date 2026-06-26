"""Local LLM narration via Ollama. The model only explains numbers it is given —
it never computes or recalls figures. Degrades gracefully if Ollama is down."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests
import config as C

BUILD = "2026-06-18e"   # bump on each edit; shown in the sidebar to confirm a real restart

SYSTEM = (
    "You are a demand-response analyst for an electricity utility. "
    "Use ONLY the numbers in the provided data -- never invent, recall, or estimate a figure, and never "
    "compare to a dataset average unless that average is present in the data. Do not invent qualitative "
    "meaning for a raw input feature that has no definition below (e.g. temp_slope, std_kWh, cv) -- "
    "report its value without editorial interpretation.\n"
    "Definitions and scales:\n"
    "- tier_raw: Reliable (responds often) / Occasional (sometimes) / Non-responder (almost never) / "
    "Sparse (fewer than 3 observed events -- too little history to tier confidently; for a sparse customer "
    "say their history is too thin to assess and do NOT report flag_rate or eb_fr as a reliability verdict).\n"
    "- tier_norm: the signal-normalised observed tier. If it diverges from tier_raw, the customer behaves "
    "differently once their dominant signal is controlled for -- explain it, do not call it a contradiction.\n"
    "- p_reliable (0-1): probability the customer is CAPABLE of reliable response, predicted from the demand "
    "profile; >0.55 leans capable, <0.40 leans not.\n"
    "- eb_fr (0-1): the shrunk OBSERVED response rate -- actual past behaviour, distinct from the predicted "
    "p_reliable. Do not compare eb_fr to the 0.55 p_reliable boundary and do not conflate the two.\n"
    "- convertibility: unused capacity to convert an OCCASIONAL customer into a reliable one; >0.20 = strong, "
    "0.10-0.20 = some, <0.10 = little headroom; NEGATIVE = no headroom (at/below baseline). Interpret it only "
    "when it is actually reported.\n"
    "- cv is the input coefficient of variation, NOT convertibility -- never relabel cv (or any input "
    "feature) as convertibility.\n"
    "- For one customer, tier_raw is OBSERVED behaviour while p_reliable is predicted from the demand "
    "profile. If they diverge (e.g. a reliable customer with low p_reliable, or negative convertibility), "
    "explain it as the customer responding MORE than their profile predicts (no conversion headroom) -- "
    "not as a contradiction.\n"
    "- lever / targeting: the recommended action -- quote it VERBATIM, never substitute your own; treat a "
    "parenthetical like 'nudge (real sub-threshold response)' as a rationale for existing behaviour, not a goal.\n"
    "CONVERTIBILITY GROUND RULE: convertibility is a CONVERSION DECISION signal ONLY for occasional "
    "customers (where it estimates unused capacity to turn them reliable). For a reliable customer it is "
    "descriptive only -- they are already won, so plan around/retain them and NEVER recommend converting "
    "them or running a conversion pilot off their convertibility. For a non-responder it is descriptive too: "
    "act only if BOTH p_reliable AND convertibility are high, and then only as a small controlled pilot. "
    "If a grouped/slice result reports convertibility (or near_miss_frac) over a population not restricted to "
    "occasionals, say it mixes tiers and is conversion-actionable only for the occasional portion -- never "
    "frame a tier-blind convertibility ranking/average as a conversion-opportunity ranking.\n"
    "Action logic: 'conversion' advice applies to OCCASIONALS only. Reliable customers are already won -- "
    "recommend retaining/planning around them, not converting; never give that retain/plan action to an "
    "occasional or non-responder. Non-responders usually aren't worth chasing UNLESS p_reliable AND "
    "convertibility are both high (worth a controlled pilot, not blanket spend).\n"
    "RESULT TYPES:\n"
    "- Slice (has 'n_users'): n_users is the count matching ALL 'filters_applied' -- usually the answer. "
    "A field named mean_<col> or <func>_<col> (e.g. mean_convertibility) IS the already-computed "
    "average/mean over those n_users ('average' and 'mean' are synonyms) -- state it plainly with n_users; "
    "NEVER refuse on the grounds that an average needs more than one value. 'pct_of_total' is their share "
    "of all customers; ignore a pct that is 100 because the group was already filtered on it. Tier shares "
    "(pct_reliable/occasional/non_responder/sparse) cover all customers; 'sparse' = fewer than 3 events.\n"
    "- Grouped (has 'results'): a PRE-SORTED ranked list; each entry has 'value' (a count, or a 0-1 share "
    "when the column is is_*) and 'n_users'. List the groups in EXACTLY the given array order -- never "
    "reorder, re-rank, or alphabetize. For a Top-N/ranking question, number them 1., 2., 3. in that order. "
    "Report all returned groups, or if you summarise only the leaders say 'top K of N'. A group label "
    "containing ' / ' is a COMBINATION of two columns (e.g. 'Oslo / B_10' is a Region+signal cell) -- "
    "never restate it as a single column or call it just a region.\n"
    "- Overview/describe (has 'queryable_columns'): summarise size, tier split (with counts), regions and "
    "signals, and suggest ONE example question. Suggest/answer an example question ONLY for this overview "
    "result -- NEVER on a slice, grouped, or prospect result.\n"
    "- Prospect score (has 'predicted_tier'): report confidence FROM 'tier_confidence'/'tier_probabilities' "
    "-- if confidence is low or the top two probabilities are within ~0.05 (near tie), say the prediction "
    "is uncertain; do NOT claim 'high probability' or apply the 0.55 p_reliable boundary as a tier "
    "threshold. Quote 'targeting' verbatim. Convertibility is NOT available for a prospect -- state that, "
    "do not invent it.\n"
    "If a 'scope_note' is present, mention its caveat. If n_users is 0 (no customers match the filters), "
    "say so plainly and give NO other figures -- never invent values for an empty result. If any reported "
    "value is null/NaN, say it is not available for that group rather than reporting it as a number.\n"
    "If a 'KEY POINTS' block is provided it is accurate and already correctly ordered: base your answer on "
    "it, keep its order, and reproduce any recommendation text exactly. You MAY use its readable labels, but "
    "NEVER use a raw snake_case column name from the Data block (e.g. tier_raw, tier_norm, flag_rate, "
    "p_reliable, n_events, n_flagged) as a bullet label -- rewrite every label in plain English (e.g. "
    "'Reliability probability', 'Response rate', 'Events observed'). A single customer's own response rate "
    "(flag_rate) is THAT customer's rate -- never call it a population 'share'. Do not echo the block's "
    "structural words or these instructions verbatim.\n"
    "FORMAT your answer in two parts. First, a 1-3 sentence plain-English takeaway that directly answers the "
    "question in natural language (no field names, no jargon). Then, on the next lines, list the supporting "
    "figures as a short bulleted list -- one figure per bullet, each a brief plain-English label YOU choose "
    "followed by its value (e.g. '- Reliable share: 42%'). Use your own wording for labels; do not copy KEY "
    "POINTS headers or raw column names, and add no figure that is not in the data. For a ranked/grouped "
    "result the bullets are the numbered groups in the given order. If there is only one figure, a single "
    "sentence with the number inline is fine instead of a list. Keep it tight -- lead sentence plus only the "
    "figures that answer the question."
)


def available():
    try:
        requests.get(f"{C.LLM_HOST}/api/tags", timeout=3)
        return True
    except Exception:
        return False


_FRIENDLY = {"mean_kWh": "average kWh", "p_reliable": "predicted reliability",
             "eb_fr": "observed response rate", "flag_rate": "response rate",
             "near_miss_frac": "near-miss fraction", "temp_slope": "temperature slope",
             "earnings_total_NOK": "earnings (NOK)", "kWh": "kWh",
             "tier_raw": "tier", "tier_norm": "normalised tier", "dom_signal": "signal",
             "Region": "region", "best_signal": "best signal"}


def _friendly(col):
    """Plain-English label for a column so no raw snake_case reaches the narration."""
    if not isinstance(col, str):
        return "value"
    if col.startswith("is_"):
        return col[3:].replace("_", "-") + " share"
    return _FRIENDLY.get(col, col.replace("_", " "))


def _key_points(facts):
    """Deterministic, pre-labelled summary of a result so the model narrates from clean
    material (right order, friendly labels, no raw field names) instead of the raw fact
    dict. PURE DATA. Returns '' when no hint applies."""
    if not isinstance(facts, dict):
        return ""
    sn = f"\nnote: {facts['scope_note']}" if facts.get("scope_note") else ""

    res = facts.get("results")
    if isinstance(res, list):                                # grouped / ranked
        if not res:
            return "0 matching groups"
        m = facts.get("metric", {})
        what = _friendly(m["col"]) if m.get("col") else m.get("func", "count")
        gb = facts.get("group_by")
        gb = " & ".join(_friendly(g) for g in gb) if isinstance(gb, list) else _friendly(gb)
        val = lambda v: v if isinstance(v, (int, float)) and v == v else "n/a"
        lines = [f"{i+1}. {r['group']}: {val(r['value'])} (n={r['n_users']})"
                 for i, r in enumerate(res[:10])]
        more = f"\n... {len(res)-10} more ({len(res)} total)" if len(res) > 10 else ""
        return f"{what} by {gb}, ranked:\n" + "\n".join(lines) + more + sn

    if "predicted_tier" in facts:                            # prospect score
        rn = f"\nnote: {facts['region_note']}" if facts.get("region_note") else ""
        return (f"predicted tier: {facts['predicted_tier']}\n"
                f"confidence: {facts.get('tier_confidence', '?')} "
                f"(probabilities {facts.get('tier_probabilities')}, p_reliable {facts.get('p_reliable')})\n"
                f"recommendation: {facts.get('targeting')}\n"
                f"convertibility: {facts.get('convertibility')}" + rn)

    if "ID" in facts and "tier_raw" in facts:                # single customer
        t = facts.get("tier_raw")
        L = []
        if t == "sparse":
            L.append("note: fewer than 3 events — too little history to judge reliability")
        elif t == "reliable" and isinstance(facts.get("p_reliable"), (int, float)) and facts["p_reliable"] < 0.5:
            L.append("note: reliable by observed behaviour despite a low predicted reliability — "
                     "responds more than the profile predicts (no conversion headroom)")
        L.append(f"observed tier: {t}")
        if facts.get("tier_norm") and facts["tier_norm"] != t:
            L.append(f"signal-normalised tier: {facts['tier_norm']} (differs from observed)")
        if "flag_rate" in facts:
            ev = (f" (responded to {facts.get('n_flagged')} of {facts.get('n_events')} events)"
                  if "n_events" in facts else "")
            L.append(f"this customer's own response rate: {facts['flag_rate']}{ev}")
        if "p_reliable" in facts:
            L.append(f"predicted reliability from profile: {facts['p_reliable']}")
        if "convertibility" in facts:
            tag = ("conversion headroom" if t == "occasional"
                   else "descriptive only — not a conversion target for this tier")
            L.append(f"convertibility: {facts['convertibility']} ({tag})")
        if facts.get("lever"):
            lev = facts["lever"]
            gloss = (" — already responds a little but BELOW the reliability threshold, so a nudge "
                     "could push them over" if "sub-threshold" in lev else "")
            L.append(f"recommended action: {lev}{gloss}")
        for k, lab in [("dom_signal", "main signal"), ("mean_kWh", "average kWh"), ("Region", "region")]:
            if k in facts:
                L.append(f"{lab}: {facts[k]}")
        return "\n".join(L)

    if "queryable_columns" in facts:                         # dataset overview
        ts = facts.get("tier_sizes") or {}
        tiers = ", ".join(f"{k} {v}" for k, v in ts.items())
        return (f"total customers: {facts.get('total_users')}\n"
                f"tiers: {tiers}\n"
                f"fleet load per event: {facts.get('fleet_MW_per_event')} MW\n"
                f"regions ({len(facts.get('regions', []))}): {', '.join(facts.get('regions', []))}\n"
                f"signals: {', '.join(facts.get('signals', []))}\n"
                f"conversion cohorts: {facts.get('conversion_cohorts')}")

    if "n_users" in facts:                                   # filtered slice summary
        fa = facts.get("filters_applied", [])
        desc = ", ".join(f"{_friendly(f['col'])} {f['op']} {f['value']}" for f in fa) or "all customers"
        n = facts["n_users"]
        if n == 0:
            return f"0 customers match ({desc})"
        L = [f"{n} customers match ({desc}) — {facts.get('pct_of_total')}% of all customers"]
        on_tier = any(f.get("col") == "tier_raw" for f in fa)
        if not on_tier:
            for lab, k in [("reliable", "pct_reliable"), ("occasional", "pct_occasional"),
                           ("non-responder", "pct_non_responder"), ("sparse", "pct_sparse")]:
                if k in facts:
                    L.append(f"{lab} share: {facts[k]}%")
        for k, v in facts.items():
            if isinstance(k, str) and "_" in k and k.split("_", 1)[0] in ("mean", "sum", "min", "max"):
                L.append(f"{_friendly(k.split('_', 1)[1])} ({k.split('_', 1)[0]}): {v}")
        return "\n".join(L) + sn

    return ""


def narrate(question, facts):
    """Return a natural-language answer grounded in `facts`, or None if unavailable."""
    hint = _key_points(facts)
    if hint:                # curated hint is comprehensive -> do NOT also ship the raw dict
        user = (f"DATA (the only figures you may use — already correct and in order):\n{hint}\n\n"
                f"Question: {question}")
    else:
        user = f"Data:\n{facts}\n\nQuestion: {question}"
    body = {
        "model": C.LLM_MODEL, "stream": False, "options": {"temperature": 0.1},  # faithful > creative
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    try:
        r = requests.post(f"{C.LLM_HOST}/api/chat", json=body, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        return None


ANALYTICS_SYS = (
    "Translate a question about a customer dataset into a JSON analytics spec. Output ONLY JSON:\n"
    '{"filters": [{"col","op","value"}], "group_by": <column or null>, '
    '"metric": {"func": "count|mean|sum|min|max", "col": <numeric column; omit for count>}, '
    '"sort": "asc|desc", "limit": <int or null>}\n'
    "Use ONLY columns/operators from the schema. Categorical filters use op '==' with an EXACT "
    "allowed value (copy region names verbatim, including ø/å). Copy numeric thresholds EXACTLY as "
    "written (e.g. 'above 50' -> value 50, never 5). Add a filter for EVERY condition mentioned "
    "(a region AND a tier = two filters). "
    "'Which/what <column> has the most/least/highest/lowest ...' means group_by THAT column, never a "
    "filter on it. Never emit a filter whose value is null. "
    "Set group_by to rank/compare across groups; null otherwise. group_by may be a single column or a "
    "list of TWO columns for a cross-tab (e.g. ['Region','dom_signal']). For a tier SHARE/percentage/"
    "proportion per group, use metric mean of is_reliable / is_occasional / is_non_responder and DO NOT "
    "add a tier filter (filtering by the tier would force every group to 100%). Only a COUNT of a tier "
    "uses a tier filter. Examples:\n"
    'Q: top 5 regions with most reliable users (count) -> {"filters":[{"col":"tier_raw","op":"==","value":"reliable"}],'
    '"group_by":"Region","metric":{"func":"count"},"sort":"desc","limit":5}\n'
    'Q: top 3 regions by reliable share -> {"filters":[],"group_by":"Region",'
    '"metric":{"func":"mean","col":"is_reliable"},"sort":"desc","limit":3}\n'
    'Q: which region has the lowest average mean_kWh -> {"filters":[],"group_by":"Region",'
    '"metric":{"func":"mean","col":"mean_kWh"},"sort":"asc","limit":1}\n'
    'Q: which signal has the highest reliable share -> {"filters":[],"group_by":"dom_signal",'
    '"metric":{"func":"mean","col":"is_reliable"},"sort":"desc","limit":1}\n'
    'Q: how many users from Oslo are reliable -> {"filters":[{"col":"Region","op":"==","value":"Oslo"},'
    '{"col":"tier_raw","op":"==","value":"reliable"}],"group_by":null,"metric":{"func":"count"},"sort":"desc","limit":null}\n'
    "No prose, JSON only."
)


def extract_analytics(question, schema):
    """Ask the model for a structured analytics spec. On failure returns an empty spec
    (treated as a dataset-wide summary)."""
    import json
    body = {
        "model": C.LLM_MODEL, "stream": False, "format": "json", "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": ANALYTICS_SYS},
            {"role": "user", "content": f"Schema:\n{json.dumps(schema)}\n\nQuestion: {question}"},
        ],
    }
    try:
        r = requests.post(f"{C.LLM_HOST}/api/chat", json=body, timeout=120)
        r.raise_for_status()
        spec = json.loads(r.json()["message"]["content"])
        return spec if isinstance(spec, dict) else {"filters": []}
    except Exception:
        return {"filters": []}


ROUTE_SYS = (
    "Classify a question about an electricity demand-response CUSTOMER dataset into ONE intent. "
    'Output JSON only: {"intent": "analytics|prospect|describe"}.\n'
    "- prospect: score/predict a NEW or hypothetical customer from given feature VALUES "
    "(numbers assigned to features, e.g. 'score a customer with mean_kWh 4.2, temp_slope -0.1').\n"
    "- describe: asks what the dataset contains, what can be asked, a whole-fleet overview/summary, "
    "or what a column means.\n"
    "- analytics: anything answered from EXISTING customers -- counts, shares, averages, rankings, "
    "group comparisons, threshold filters. This is the DEFAULT when unsure.\n"
    "JSON only."
)


def classify(question):
    """Route a non-user question to 'analytics' (default), 'prospect', or 'describe'."""
    import json
    body = {
        "model": C.LLM_MODEL, "stream": False, "format": "json", "options": {"temperature": 0.0},
        "messages": [{"role": "system", "content": ROUTE_SYS},
                     {"role": "user", "content": question}],
    }
    try:
        r = requests.post(f"{C.LLM_HOST}/api/chat", json=body, timeout=60)
        r.raise_for_status()
        intent = str(json.loads(r.json()["message"]["content"]).get("intent", "analytics"))
        return intent if intent in ("analytics", "prospect", "describe") else "analytics"
    except Exception:
        return "analytics"


def extract_features(question, feats):
    """Pull demand-feature values out of the question into {feature: number}.
    Returns {} on failure."""
    import json
    sys_p = ("Extract demand-feature values from the question into JSON mapping feature "
             "names to numbers, plus 'Region' as a string if mentioned. "
             f"Allowed features: {feats}. Example: "
             '{"mean_kWh": 3.1, "temp_slope": -0.09}. Output JSON only, no prose.')
    body = {
        "model": C.LLM_MODEL, "stream": False, "format": "json", "options": {"temperature": 0.0},
        "messages": [{"role": "system", "content": sys_p},
                     {"role": "user", "content": question}],
    }
    try:
        r = requests.post(f"{C.LLM_HOST}/api/chat", json=body, timeout=120)
        r.raise_for_status()
        d = json.loads(r.json()["message"]["content"])
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

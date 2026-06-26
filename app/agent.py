"""One entry point for the single query bar. A user ID is matched deterministically;
every other question is routed by a lightweight LLM intent classifier to dataset
analytics, prospect scoring, or a dataset overview. Python owns every number; the
LLM only routes and narrates."""
import re
import services, llm

_ID = re.compile(r"Exp_\d+", re.IGNORECASE)


def answer(question):
    """Return (prose, ground_truth_dict)."""
    q = question or ""
    ids = _ID.findall(q)
    if ids:                                   # ---- per-user lookup (deterministic) ----
        uid = ids[0].split("_")[0].title() + "_" + ids[0].split("_")[1]
        facts = services.analyze_user(uid)
        if facts is None:
            return f"Unknown user '{uid}'.", None
        prose = llm.narrate(question, facts)
        if prose and len({i.lower() for i in ids}) > 1:   # several IDs named; only the first is shown
            prose += f"\n\n_(Showing {uid}; you also named other IDs — ask about them one at a time.)_"
        return prose, facts

    intent = llm.classify(q)                   # ---- route the rest ----
    if intent == "prospect":                   # score a new/hypothetical customer
        vals = llm.extract_features(q, services.feature_names())
        result = services.score_features(vals)
    elif intent == "describe":                 # what's in the data / overview
        result = services.describe()
    else:                                       # analytics: filter / aggregate / rank
        spec = llm.extract_analytics(q, services.query_schema())
        result = services.run_analytics(spec)
    return llm.narrate(question, result), result

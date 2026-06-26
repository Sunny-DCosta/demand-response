"""Deterministic data/model access layer. Loads pipeline artifacts once; the
dashboard and the LLM both call these. No model math happens in the LLM."""
import sys, json, functools, operator, unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd, joblib
import config as C


def _fix_enc(s):
    try: return s.encode("latin-1").decode("utf-8")   # repair Bodø/Tromsø double-encoding
    except Exception: return s


@functools.lru_cache(maxsize=1)
def _users():
    df = pd.read_csv(C.TIERS, dtype={"ID": str})
    for f in (C.SCORES, C.CONVERSION):
        if Path(f).exists():
            df = df.merge(pd.read_csv(f, dtype={"ID": str}), on="ID", how="left")
    if "Region" in df.columns:
        df["Region"] = df["Region"].map(_fix_enc)
    if "tier_raw" in df.columns:                       # 0/1 indicators -> mean() = tier share
        for t in ["reliable", "occasional", "non-responder", "sparse"]:
            df[f"is_{t.replace('-', '_')}"] = (df["tier_raw"] == t).astype(int)
        # conversion diagnostics are meaningful ONLY for occasionals; null them elsewhere so no
        # consumer (per-user, slice, or group-by) can surface a 'convert' signal for other tiers
        not_occ = df["tier_raw"] != "occasional"
        for c in ("lever", "near_miss_frac"):
            if c in df.columns:
                df.loc[not_occ, c] = None
    return df.set_index("ID")


@functools.lru_cache(maxsize=1)
def cohort_summary():
    return json.loads(Path(C.SUMMARY).read_text())


@functools.lru_cache(maxsize=1)
def _model():
    return joblib.load(C.RF_PKL) if Path(C.RF_PKL).exists() else None


def all_user_ids():
    return _users().index.tolist()


def analyze_user(uid):
    """Return the curated fact dict for one user, or None if unknown."""
    u = _users()
    uid = str(uid).strip()
    if uid not in u.index:
        return None
    r = u.loc[uid]
    keep = ["tier_raw", "tier_norm", "flag_rate", "n_events", "n_flagged", "p_reliable",
            "convertibility", "dom_signal", "best_signal", "near_miss_frac", "lever",
            "temp_slope", "mean_kWh", "Region"]
    out = {"ID": uid}
    for k in keep:
        if k in u.columns and pd.notna(r[k]):
            v = r[k].item() if hasattr(r[k], "item") else r[k]   # numpy int64/float64 -> native
            out[k] = round(v, 3) if isinstance(v, float) else v
    if out.get("tier_raw") != "occasional":     # conversion diagnostics apply only to occasionals;
        out.pop("lever", None)                  # a reliable/non-responder has no 'convert' lever
        out.pop("near_miss_frac", None)
    return out


def find_convertible(n=15):
    """Top occasional conversion candidates within the actionable cohort."""
    u = _users().reset_index()
    if "lever" not in u.columns:
        return pd.DataFrame()
    act = u[(u["tier_raw"] == "occasional") & u["lever"].str.startswith(("reassign", "nudge"), na=False)]
    cols = [c for c in ["ID", "dom_signal", "n_events", "eb_fr", "p_reliable",
                        "convertibility", "near_miss_frac", "lever"] if c in act.columns]
    return act.sort_values("convertibility", ascending=False)[cols].head(n).round(3)


def feature_names():
    b = _model()
    return b["feats"] if b else C.BASE_FEATS


def _recommend(tier, p):
    if tier == "reliable" or p >= 0.55:
        return "TARGET — strong demand profile; worth enrolling/prioritising."
    if tier == "occasional":
        return "MAYBE — partial capacity; a nudge candidate if p_reliable is moderate-high, else low priority."
    return "SKIP — low predicted capacity; unlikely to respond to incentives."


def score_features(features: dict):
    """Score a hypothetical/new customer from the 6 demand features (+ optional Region).
    Returns predicted tier, reliability probability and a targeting call. Convertibility
    is NOT returned: it needs observed response history, which a prospect has none of."""
    b = _model()
    if b is None:
        return None
    row = {f: float(features.get(f, 0) or 0) for f in b["feats"]}
    X = pd.DataFrame([row]).reindex(columns=b["cols"], fill_value=0)
    reg = features.get("Region")
    if reg and f"reg_{reg}" in b["cols"]:
        X[f"reg_{reg}"] = 1
    p = float(b["model"].predict_proba(X)[0, 1])
    out = {"input_features": row, "p_reliable": round(p, 3)}
    if "model3" in b:
        proba = b["model3"].predict_proba(X)[0]
        out["predicted_tier"] = str(b["model3"].classes_[proba.argmax()])
        out["tier_probabilities"] = {str(c): round(float(pp), 3)
                                     for c, pp in zip(b["model3"].classes_, proba)}
        top = sorted((float(pp) for pp in proba), reverse=True)      # confidence from the data,
        out["tier_confidence"] = ("low" if len(top) > 1 and top[0] - top[1] < 0.05  # so the narrator
                                  else "high" if top[0] >= 0.55 else "moderate")     # need not guess
    else:
        out["predicted_tier"] = "reliable" if p >= 0.5 else "not-reliable"
    out["targeting"] = _recommend(out["predicted_tier"], p)
    out["convertibility"] = "not available for a prospect — needs observed response history"
    if not reg:
        out["region_note"] = ("no Region supplied; scored at baseline region — "
                               "add a Region (e.g. Oslo, Bodø) for an accurate score")
    return out


# ── generic query layer (schema auto-derived from the live data) ─────────────
_OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge,
        "<=": operator.le, "==": operator.eq, "!=": operator.ne}
_HIDE = {"ID", "cluster"}            # identifiers / not meaningful to query directly


@functools.lru_cache(maxsize=1)
def _schema():
    """Derive queryable columns from the dataframe so new pipeline columns become
    askable automatically. numeric = anything aggregatable; categorical =
    low-cardinality text (with its allowed values)."""
    u = _users().reset_index()
    numeric, categ = [], {}
    for c in u.columns:
        if c in _HIDE:
            continue
        s = u[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
        elif s.nunique(dropna=True) <= 40:
            categ[c] = sorted(map(str, s.dropna().unique()))
    return numeric, categ


def query_schema():
    """Columns + allowed values, handed to the LLM so it maps a question to a spec."""
    num, cat = _schema()
    return {"numeric_columns": num, "categorical_columns": cat, "operators": list(_OPS),
            "share_columns": [c for c in num if c.startswith("is_")]}


_NORDIC = str.maketrans({"ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "å": "a", "Å": "a"})


def _fold(x):
    """Normalise a categorical value so LLM variants match the stored one:
    'Tromso'/'TROMSØ' -> 'Tromsø', 'non_responder' -> 'non-responder'. Maps Nordic
    letters (NFKD handles å but not ø/æ), casefolds, and drops spaces/-/_ separators."""
    s = unicodedata.normalize("NFKD", str(x)).translate(_NORDIC)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return "".join(ch for ch in s if ch.isalnum())


def _apply_filters(df, filters):
    num, cat = _schema()
    applied, skipped = [], []
    for f in filters or []:
        col, op, val = f.get("col"), f.get("op"), f.get("value")
        if val is None or op not in _OPS or col not in df.columns or col not in num + list(cat):
            skipped.append(f); continue
        try:
            if col in cat:
                if op not in ("==", "!="): skipped.append(f); continue
                # resolve a mis-encoded / mis-cased value to the real category
                use = next((a for a in cat[col] if _fold(a) == _fold(val)), val)
                df = df[_OPS[op](df[col].astype(str), str(use))]
                applied.append({"col": col, "op": op, "value": use})
            else:
                df = df[_OPS[op](df[col].astype(float), float(val))]
                applied.append({"col": col, "op": op, "value": val})
        except Exception:
            skipped.append(f)
    return df, applied, skipped


def run_analytics(spec):
    """General query: filter rows, optionally group-by (1-2 columns) + aggregate + sort + limit.
    spec = {filters:[{col,op,value}], group_by, metric:{func,col}, sort:'asc'|'desc', limit}.
    Without group_by it summarises the filtered slice; with it, returns ranked groups."""
    num, cat = _schema()
    valid = num + list(cat)
    metric = spec.get("metric") or {"func": "count"}
    func = str(metric.get("func", "count")).lower()
    col = metric.get("col")

    # convertibility/near_miss_frac are conversion signals -> only meaningful over occasionals.
    # If such a metric is aggregated without a tier filter, scope it to occasionals (and say so).
    filters = list(spec.get("filters") or [])
    out = {}
    if (col in ("convertibility", "near_miss_frac")
            and not any(f.get("col") == "tier_raw" for f in filters)):
        filters.append({"col": "tier_raw", "op": "==", "value": "occasional"})
        out["scope_note"] = f"{col} is a conversion signal — scoped to occasional customers only"

    u = _users()
    df, applied, skipped = _apply_filters(u.reset_index(), filters)
    out["filters_applied"] = applied
    if skipped:
        out["filters_skipped"] = skipped

    gb = spec.get("group_by")
    gbs = [gb] if isinstance(gb, str) else list(gb or [])
    gbs = [g for g in gbs if g in valid and g in df.columns]

    if gbs and len(df):
        g = df.groupby(gbs[0] if len(gbs) == 1 else gbs)
        cnt = g.size()
        if func in ("mean", "sum", "min", "max") and col in num and col in df.columns:
            s = getattr(g[col], func)(); used = {"func": func, "col": col}
        else:
            s = cnt; used = {"func": "count"}
        s = s.sort_values(ascending=str(spec.get("sort", "desc")).lower() == "asc")
        lim = spec.get("limit")
        if isinstance(lim, (int, float)) and lim > 0:
            s = s.head(int(lim))
        key = lambda k: " / ".join(map(str, k)) if isinstance(k, tuple) else str(k)
        out.update({"group_by": gbs if len(gbs) > 1 else gbs[0], "metric": used,
                    "results": [{"group": key(k), "value": round(float(v), 3),
                                 "n_users": int(cnt[k])} for k, v in s.items() if pd.notna(v)]})
        return out

    # no grouping -> slice summary
    n, tot = len(df), len(u)
    out.update({"n_users": n, "pct_of_total": round(100*n/tot, 1) if tot else 0})
    if n:
        for t in ["reliable", "occasional", "non-responder", "sparse"]:
            out[f"pct_{t.replace('-', '_')}"] = round(100*(df["tier_raw"] == t).mean(), 1)
        if func in ("mean", "sum", "min", "max") and col in num and col in df.columns:
            v = getattr(df[col], func)()
            if pd.notna(v):
                out[f"{func}_{col}"] = round(float(v), 3)
        out["means"] = {c: round(float(df[c].mean()), 3)
                        for c in ["flag_rate", "p_reliable", "mean_kWh", "temp_slope"]
                        if c in df.columns and pd.notna(df[c].mean())}
    return out


def describe():
    """High-level overview of the dataset + what can be asked (the 'describe' intent)."""
    s = cohort_summary()
    num, cat = _schema()
    return {
        "total_users": s.get("total_users"),
        "tier_sizes": s["tier_sizes"]["raw"],
        "fleet_MW_per_event": s["load_shed"]["fleet_unbiased_mw"],
        # regions from the repaired dataframe (summary.json keys are mojibake-encoded)
        "regions": sorted(_users()["Region"].dropna().unique().tolist()),
        "signals": list(s.get("by_signal", {}).keys()),
        "conversion_cohorts": {k: v for k, v in (s.get("conversion", {}).get("cohorts") or {}).items()
                               if "reassign" not in k.lower()},   # reassign cohort hidden per request
        "queryable_columns": {"numeric": num, "categorical": list(cat)},
    }

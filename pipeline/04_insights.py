"""Stage 4: business insights for the dashboard.
Outputs:
  insights/summary.json  -- tier sizes, dose-response, signal-bias chi2, load-shed, cohort sizes
  data/conversion.csv    -- per-user near-miss + best-signal + recommended lever
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
import config as C


def lever(r):
    if pd.notna(r["best_signal_fr"]) and r["best_signal_fr"] >= 0.50 and r["best_signal"] != r["dom_signal"]:
        return f"reassign signal -> {r['best_signal']}"
    if pd.notna(r["near_miss_frac"]) and r["near_miss_frac"] >= 0.30:
        return "nudge (real sub-threshold response)"
    return "needs pilot (no usable single-customer signal)"


def main():
    ev = pd.read_csv(C.EVENTS, dtype={"ID": str})
    u  = pd.read_csv(C.TIERS, dtype={"ID": str})
    sc = pd.read_csv(C.SCORES, dtype={"ID": str})
    u  = u.merge(sc, on="ID", how="left")
    ev["profile"] = ev["signal"].map(C.profile_of)
    out = {}

    # ---- tier sizes ----
    out["tier_sizes"] = {
        "raw":  u["tier_raw"].value_counts().to_dict(),
        "norm": u["tier_norm"].value_counts().to_dict(),
    }

    # ---- dose-response per signal ----
    by = ev.groupby("signal").agg(n=("flagged", "size"), flag_rate=("flagged", "mean"))
    rd = ev[ev["flagged"]].groupby("signal")["rel_drop"].mean()
    dose = by.join(rd).reset_index()
    dose["profile"] = dose["signal"].map(C.profile_of)
    out["dose_response"] = dose.round(3).to_dict("records")

    # ---- signal bias (chi2 on dominant-signal family x raw tier) ----
    main_u = u[u["tier_raw"] != "sparse"].copy()
    main_u["fam"] = main_u["dom_signal"].map(C.profile_of)
    ct = pd.crosstab(main_u["fam"], main_u["tier_raw"])
    chi2, p, dof, _ = chi2_contingency(ct)
    out["signal_bias"] = {"chi2": round(float(chi2), 1), "p": float(p), "dof": int(dof)}

    # ---- load-shed: conditional vs unbiased ----
    ev["kwh_shed"] = (-ev["mean_peak_resid"]) * ev["n_peak_hours"]
    evt = ev.merge(u[["ID", "tier_raw"]], on="ID", how="left")
    nu = main_u.groupby("tier_raw").size()
    shed = []
    fc = fu = 0.0
    for t in ["reliable", "occasional", "non-responder"]:
        sub = evt[evt["tier_raw"] == t]; n = int(nu.get(t, 0))
        cond = float((sub.loc[sub["flagged"], "kwh_shed"].mean() or 0) * sub["flagged"].mean())
        unb  = float(sub["kwh_shed"].mean())
        fc += cond*n; fu += unb*n
        shed.append({"tier": t, "users": n, "kwh_ev_conditional": round(cond, 3),
                     "kwh_ev_unbiased": round(unb, 3)})
    out["load_shed"] = {"per_tier": shed, "fleet_cond_mw": round(fc/1000, 2),
                        "fleet_unbiased_mw": round(fu/1000, 2)}

    # ---- breakdowns by region / dominant signal (for general questions) ----
    evt2 = ev.merge(u[["ID", "Region", "dom_signal"]], on="ID", how="left")

    def breakdown(col):
        tot = len(main_u); rows = {}
        for key, g in main_u.groupby(col):
            evg = evt2[evt2[col] == key]
            rows[str(key)] = {
                "n_users": int(len(g)),
                "pct_of_total": round(100*len(g)/tot, 1),
                "pct_reliable": round(100*(g["tier_raw"] == "reliable").mean(), 1),
                "pct_occasional": round(100*(g["tier_raw"] == "occasional").mean(), 1),
                "pct_non_responder": round(100*(g["tier_raw"] == "non-responder").mean(), 1),
                "fleet_kwh_per_event": round(float(evg["kwh_shed"].mean() * len(g)), 1),
            }
        return rows

    out["total_users"] = int(len(main_u))
    out["by_region"] = breakdown("Region")
    out["by_signal"] = breakdown("dom_signal")

    # ---- conversion diagnostic (occasionals) ----
    ev["flagged_i"] = ev["flagged"].astype(int)
    sf = ev.groupby(["ID", "signal"]).agg(fr=("flagged_i", "mean"), n=("flagged_i", "size"))
    sf = sf[sf["n"] >= 2]
    best_fr  = sf.groupby("ID")["fr"].max().rename("best_signal_fr")
    best_sig = sf["fr"].groupby("ID").idxmax().map(lambda x: x[1]).rename("best_signal")
    near = (ev[~ev["flagged"]].assign(nm=lambda d: d["rel_drop"].between(0.05, 0.10))
              .groupby("ID")["nm"].mean().rename("near_miss_frac"))

    conv = u[["ID", "dom_signal", "n_events", "eb_fr", "tier_raw"]].merge(
        sc, on="ID", how="left").join(
        pd.concat([best_fr, best_sig, near], axis=1), on="ID")
    occ = conv[conv["tier_raw"] == "occasional"].copy()
    occ["lever"] = occ.apply(lever, axis=1)

    actionable = occ["lever"].str.startswith(("reassign", "nudge"))
    out["conversion"] = {
        "n_occasional": int(len(occ)),
        "cohorts": occ["lever"].value_counts().to_dict(),
        "actionable": int(actionable.sum()),
        "high_convertibility": int((occ["convertibility"] > 0.20).sum()),
    }

    # per-user conversion table (for the dashboard user lookup)
    conv["lever"] = conv.apply(lever, axis=1)
    conv[["ID", "near_miss_frac", "best_signal", "best_signal_fr", "lever"]].to_csv(C.CONVERSION, index=False)

    C.SUMMARY.write_text(json.dumps(out, indent=2))
    print(f"[4] insights -> {C.SUMMARY.name}  + {C.CONVERSION.name}  | "
          f"actionable occasionals: {int(actionable.sum())}/{len(occ)}")


if __name__ == "__main__":
    main()

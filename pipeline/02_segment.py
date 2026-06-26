"""Stage 2: flag_rate -> Reliable/Occasional/Non-responder tiers, plus the two
de-biasing corrections (signal-difficulty normalisation + empirical-Bayes shrink).
Output: data/user_tiers.csv (one row per user, with features for the targeting model)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import config as C


def tier(fr, n):
    if n < C.MIN_EVENTS: return "sparse"
    return ("reliable" if fr >= C.RELIABLE_THR else
            "occasional" if fr >= C.OCCASIONAL_THR else "non-responder")


def main():
    ev = pd.read_csv(C.EVENTS, dtype={"ID": str})
    feats = ["ID"] + C.BASE_FEATS + ["Region", "cluster", "earnings_total_NOK"]
    ue = pd.read_csv(C.EARNINGS, dtype={"ID": str})[feats]

    agg = ev.groupby("ID").agg(n_events=("flagged", "size"), n_flagged=("flagged", "sum")).reset_index()
    agg["flag_rate"] = agg["n_flagged"] / agg["n_events"]
    u = agg.merge(ue, on="ID", how="left")

    # signal difficulty -> normalise; Beta(2,2) -> shrink coarse n=3-5 rates
    sig_rate = ev.groupby("signal")["flagged"].mean()
    u["expected_rate"] = u["ID"].map(ev.assign(sr=ev["signal"].map(sig_rate)).groupby("ID")["sr"].mean())
    overall = float(ev["flagged"].mean())
    u["adj_fr"] = (u["flag_rate"] - u["expected_rate"] + overall).clip(0, 1)
    u["eb_fr"] = (u["n_flagged"] + 2) / (u["n_events"] + 4)
    u["dom_signal"] = u["ID"].map(ev.sort_values("event_time").groupby("ID")["signal"]
                                    .agg(lambda s: s.value_counts().index[0]))

    u["tier_raw"]  = [tier(f, n) for f, n in zip(u["flag_rate"], u["n_events"])]
    u["tier_norm"] = [tier(f, n) for f, n in zip(u["adj_fr"],   u["n_events"])]

    u.to_csv(C.TIERS, index=False)
    counts = u["tier_raw"].value_counts()
    print(f"[2] tiers -> {C.TIERS.name}  | " +
          "  ".join(f"{t}:{counts.get(t,0)}" for t in ["reliable", "occasional", "non-responder", "sparse"]))


if __name__ == "__main__":
    main()

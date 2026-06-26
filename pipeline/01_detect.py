"""Stage 1: LSTM v5 residuals -> signal-specific peaks -> 5 statistical fixes -> events.
Heavy step (LSTM forward pass ~3 min first run); per-row residuals are cached so
reruns are fast. Output: data/events_v2_signal.csv (with event_time + temp).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import config as C

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG = np.random.default_rng(C.SEED)


class LSTMRegressor(nn.Module):
    def __init__(self, n, h, L, d):
        super().__init__()
        self.lstm = nn.LSTM(n, h, num_layers=L, batch_first=True, dropout=d if L > 1 else 0.0)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def compute_test_residuals():
    tsa = pd.read_parquet(C.TSA); tsa["ID"] = tsa["ID"].astype(str)
    parts = pd.read_csv(C.DATA / "participants.csv", usecols=["ID", "Region"], encoding="latin-1")
    parts["ID"] = parts["ID"].astype(str)
    tsa = tsa.merge(parts, on="ID", how="left"); tsa["Region"] = tsa["Region"].fillna("UNK")
    clu = pd.read_csv(C.DATA / "user_clusters_k2.csv", index_col=0); clu.index = clu.index.astype(str)
    tsa = tsa.merge(clu, left_on="ID", right_index=True, how="inner"); tsa["cluster"] = tsa["cluster"].astype(np.int8)
    hourly = pd.read_csv(C.DATA / "data_hourly.csv",
                         usecols=["ID", "From", "Temperature", "Temperature24", "Temperature48", "Temperature72"],
                         parse_dates=["From"]); hourly["ID"] = hourly["ID"].astype(str)
    tsa = (tsa.merge(hourly, on=["ID", "From"], how="left")
              .dropna(subset=["Temperature"]).sort_values(["ID", "From"]).reset_index(drop=True))
    non = tsa[tsa["event_flag"] == 0]
    hm = non.groupby(["ID", "Hour"], observed=True)["Demand_kWh"].mean().reset_index().rename(columns={"Demand_kWh": "hm"})
    tsa = tsa.merge(hm, on=["ID", "Hour"], how="left")
    tsa["hm"] = tsa["hm"].fillna(tsa["ID"].map(non.groupby("ID")["Demand_kWh"].mean()))
    tsa["Demand_imp"] = np.where(tsa["event_flag"] == 1, tsa["hm"].astype(np.float32), tsa["Demand_kWh"].astype(np.float32))
    tsa.drop(columns="hm", inplace=True)
    print(f"  rows {len(tsa):,}  users {tsa['ID'].nunique():,}")

    reg = pd.get_dummies(tsa["Region"], prefix="r", drop_first=True).astype(np.float32)
    region_cols = reg.columns.tolist(); tsa = pd.concat([tsa, reg], axis=1)
    users = tsa["ID"].drop_duplicates().tolist(); u2i = {u: i for i, u in enumerate(users)}
    arrays = [None]*len(users); eflags = [None]*len(users); psig = [None]*len(users)
    horig = [None]*len(users); acts = [None]*len(users)
    mu_log = np.zeros(len(users), np.float32); sg_log = np.ones(len(users), np.float32)
    tcols = ["Temperature", "Temperature24", "Temperature48", "Temperature72"]
    tsum = np.zeros(4); tsq = np.zeros(4); tn = 0
    for uid, g in tsa.groupby("ID", observed=True, sort=False):
        i = u2i[uid]; g = g.sort_values("From"); T = len(g); tr_end = max(C.LOOKBACK+1, int(0.7*T))
        d = g["Demand_imp"].to_numpy(np.float32); dl = np.log1p(d); acts[i] = g["Demand_kWh"].to_numpy(np.float32)
        m = dl[:tr_end].mean(); s = dl[:tr_end].std()+1e-6; mu_log[i] = m; sg_log[i] = s; dn = (dl-m)/s
        hour = g["Hour"].to_numpy()-1; dow = g["From"].dt.dayofweek.to_numpy(); mon = g["From"].dt.month.to_numpy()
        tr = g[tcols].to_numpy(np.float32); tsum += tr[:tr_end].sum(0); tsq += (tr[:tr_end]**2).sum(0); tn += tr_end
        feat = np.column_stack([
            dn, np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
            np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
            np.sin(2*np.pi*mon/12), np.cos(2*np.pi*mon/12), (dow >= 5).astype(np.float32),
            tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3], (tr[:, 0]-tr[:, 1]), (tr[:, 0]-tr[:, 3]),
            g[region_cols].to_numpy(np.float32), np.full(T, int(g["cluster"].iloc[0]), np.float32),
        ]).astype(np.float32)
        arrays[i] = feat; eflags[i] = g["event_flag"].to_numpy(np.int8)
        psig[i] = g["Price_signal"].to_numpy(); horig[i] = g["Hour"].to_numpy(np.int8)
    t_mu = tsum/tn; t_sg = np.sqrt(tsq/tn - t_mu**2)+1e-6
    for i in range(len(users)):
        for j, c in enumerate([8, 9, 10, 11]):
            arrays[i][:, c] = (arrays[i][:, c]-t_mu[j])/t_sg[j]

    test_pairs = []
    for i in range(len(arrays)):
        T = len(arrays[i])
        for t in range(max(C.LOOKBACK, int(0.8*T)), T): test_pairs.append((i, t))
    print(f"  test rows: {len(test_pairs):,}")

    model = LSTMRegressor(C.N_FEATURES, C.HIDDEN, C.NUM_LAYERS, C.DROPOUT).to(DEVICE)
    ck = torch.load(str(C.LSTM_WEIGHTS), map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    n = len(test_pairs); preds_n = np.empty(n, np.float32); ui = np.empty(n, np.int32); ti = np.empty(n, np.int32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, C.BATCH):
            b = test_pairs[s:s+C.BATCH]
            X = torch.from_numpy(np.stack([arrays[u][t-C.LOOKBACK:t] for u, t in b])).to(DEVICE)
            preds_n[s:s+len(b)] = model(X).cpu().numpy()
            for j, (u, t) in enumerate(b): ui[s+j] = u; ti[s+j] = t
    preds = np.clip(np.expm1(preds_n*sg_log[ui]+mu_log[ui]), 0, None).astype(np.float32)
    acts_t = np.array([acts[u][t] for u, t in test_pairs], np.float32)
    print(f"  predicted in {time.time()-t0:.0f}s  test RMSE = {np.sqrt(np.mean((preds-acts_t)**2)):.4f}")

    idx = tsa[["ID", "From", "Temperature"]].sort_values(["ID", "From"]).copy()
    idx["t"] = idx.groupby("ID", observed=True).cumcount()
    td = pd.DataFrame({
        "ID": np.array([users[u] for u in ui]), "t": ti, "pred": preds, "resid": acts_t-preds,
        "event_flag": np.array([eflags[u][t] for u, t in test_pairs], np.int8),
        "hour": np.array([horig[u][t] for u, t in test_pairs], np.int8),
        "signal": np.array([psig[u][t] for u, t in test_pairs], dtype=object),
    }).merge(idx, on=["ID", "t"], how="left")
    pc = pd.read_csv(C.DATA / "participants.csv", usecols=["ID", "Control_Price_Phase2"], encoding="latin-1")
    pc["ID"] = pc["ID"].astype(str)
    ctrl = set(pc.loc[pc["Control_Price_Phase2"] == "Control group", "ID"])
    td["is_control"] = td["ID"].isin(ctrl).astype(np.int8)
    return td


def main():
    if C.RESID_CACHE.exists():
        td = pd.read_parquet(C.RESID_CACHE); print(f"[1] loaded cached residuals {td.shape}")
    else:
        print("[1] computing LSTM residuals ...")
        td = compute_test_residuals(); td.to_parquet(C.RESID_CACHE)

    # signal-specific peak hours
    ps = pd.read_csv(C.DATA / "price_signals.csv")
    thr = {sig: (0.48 if sig == "C" else 0.70) for sig in ps["Price_signal"].unique()}
    peaks = {}
    for sig, g in ps.groupby("Price_signal"):
        peaks[sig] = set(g.loc[g["Experiment_price_NOK_kWh"] >= g["Experiment_price_NOK_kWh"].max()*thr[sig], "Hour"].astype(int))

    td = td.sort_values(["ID", "t"]).reset_index(drop=True)
    td["is_sig_peak"] = 0
    for sig, pk in peaks.items():
        td.loc[(td["event_flag"] == 1) & (td["signal"] == sig) & (td["hour"].isin(pk)), "is_sig_peak"] = 1

    # [4] DiD
    cm = td.loc[td["is_control"] == 1].groupby("From")["resid"].mean().rename("ctrl")
    td = td.merge(cm, on="From", how="left"); td["ctrl"] = td["ctrl"].fillna(0.0)
    td["r_did"] = td["resid"] - td["ctrl"]
    # [3] Jensen recenter
    ne = td[(td["is_control"] == 0) & (td["event_flag"] == 0)]
    um = ne.groupby("ID")["r_did"].mean(); gm = float(ne["r_did"].mean())
    td["r_adj"] = td["r_did"] - td["ID"].map(um).fillna(gm)
    us = ne.assign(r_adj=td.loc[ne.index, "r_adj"]).groupby("ID")["r_adj"].agg(s="std", n="size")
    g_std = float(td.loc[ne.index, "r_adj"].std())
    us.loc[us["n"] < C.MIN_NON_EVENT, "s"] = g_std
    std_adj = us["s"].fillna(g_std).to_dict()

    # rebuild contiguous events on treated rows
    tr = td[td["is_control"] == 0].sort_values(["ID", "t"]).reset_index(drop=True)
    e = tr["event_flag"].to_numpy(); t = tr["t"].to_numpy()
    new_u = (tr["ID"] != tr["ID"].shift()).to_numpy()
    pe = np.r_[0, e[:-1]]; pt = np.r_[t[0]-999, t[:-1]]
    tr["eid"] = ((e == 1) & ((pe == 0) | new_u | (t-pt > 1))).cumsum() * e
    tr.loc[e == 0, "eid"] = -1

    pk = tr[(tr["eid"] >= 1) & (tr["is_sig_peak"] == 1)]
    ev = (pk.groupby("eid").agg(ID=("ID", "first"), signal=("signal", "first"),
                                n_peak_hours=("r_adj", "size"), mean_peak_resid=("r_adj", "mean"),
                                mean_peak_pred=("pred", "mean"), event_time=("From", "first"),
                                temp=("Temperature", "mean")).reset_index())
    ev = ev[ev["n_peak_hours"] >= C.MIN_PEAK_HOURS].copy()
    ev["user_std"] = ev["ID"].map(std_adj).fillna(g_std)
    ev["z"] = ev["mean_peak_resid"] / (ev["user_std"] / np.sqrt(ev["n_peak_hours"]))
    ev["p"] = norm.cdf(ev["z"])
    ev["reject_fdr"], ev["q"], _, _ = multipletests(ev["p"].to_numpy(), alpha=C.ALPHA_FDR, method="fdr_bh")
    ev["rel_drop"] = -ev["mean_peak_resid"] / ev["mean_peak_pred"]
    ev["flagged"] = ev["reject_fdr"] & (ev["rel_drop"] >= C.MIN_REL_DROP)
    ev.to_csv(C.EVENTS, index=False)
    print(f"[1] flagged {int(ev['flagged'].sum()):,}/{len(ev):,} ({100*ev['flagged'].mean():.1f}%) "
          f"-> {C.EVENTS.name}")


if __name__ == "__main__":
    main()

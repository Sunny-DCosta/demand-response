"""Stage 3: targeting model. Predict P(reliable) from 6 demand features + region
(no signal flag-rates -> no leakage). Stores out-of-fold scores for existing users
(honest) and a full-fit model for scoring new enrolees.
Outputs: models/rf_reliable.pkl, data/user_scores.csv (p_reliable, convertibility)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
import config as C


def build_X(df, cols=None):
    X = pd.concat([df[C.BASE_FEATS], pd.get_dummies(df["Region"], prefix="reg", drop_first=True)], axis=1).astype(float)
    return X.reindex(columns=cols, fill_value=0) if cols is not None else X


def main():
    u = pd.read_csv(C.TIERS, dtype={"ID": str})
    m = u[u["tier_raw"] != "sparse"].dropna(subset=C.BASE_FEATS).copy()
    X = build_X(m); y = (m["tier_raw"] == "reliable").astype(int)

    rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=C.SEED, n_jobs=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    p = cross_val_predict(rf(), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, p)
    order = np.argsort(p)[::-1][:len(p)//10]
    print(f"[3] RF reliable-vs-rest: AUC {auc:.3f} | top-decile lift {y.iloc[order].mean()/y.mean():.2f}x")

    final = rf().fit(X, y)                       # binary (reliable vs rest) for p_reliable
    rf3 = rf().fit(X, m["tier_raw"])             # 3-class for predicted-tier of new users
    joblib.dump({"model": final, "model3": rf3, "classes3": list(rf3.classes_),
                 "cols": X.columns.tolist(), "feats": C.BASE_FEATS}, C.RF_PKL)

    m["p_reliable"] = p
    m["convertibility"] = m["p_reliable"] - m["eb_fr"]
    m[["ID", "p_reliable", "convertibility"]].to_csv(C.SCORES, index=False)
    print(f"[3] saved {C.RF_PKL.name} + {C.SCORES.name} ({len(m)} users)")


if __name__ == "__main__":
    main()

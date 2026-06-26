"""Single source of truth for the demand-response pipeline + dashboard.
Place the iFlex-derived input files in ./data (git-ignored); the pipeline writes
models/ and insights/ alongside this file."""
from pathlib import Path

ROOT     = Path(__file__).resolve().parent          # repo root
DATA     = ROOT / "data"                             # bring your own data here (git-ignored)
MODELS   = ROOT / "models"
INSIGHTS = ROOT / "insights"
LOGO     = ROOT / "logo.png"                         # brand logo for the dashboard
BITSY    = ROOT / "bitsy.png"                         # chat avatar (user + assistant)
MODELS.mkdir(exist_ok=True)
INSIGHTS.mkdir(exist_ok=True)

BRAND    = "#2369D2"                                 # OnPoint blue (sampled from logo)

# ---- artifacts (the contract between stages) --------------------------------
LSTM_WEIGHTS = DATA / "lstm_v5_best.pt"
TSA          = DATA / "tsa_baseline.parquet"
RESID_CACHE  = DATA / "lstm_v5_test_residuals.parquet"
EVENTS       = DATA / "events_v2_signal.csv"
EARNINGS     = DATA / "per_user_earnings.csv"      # upstream: Block-A demand features
TIERS        = DATA / "user_tiers.csv"
SCORES       = DATA / "user_scores.csv"
CONVERSION   = DATA / "conversion.csv"
RF_PKL       = MODELS / "rf_reliable.pkl"
SUMMARY      = INSIGHTS / "summary.json"

# ---- LSTM v5 hyperparameters (MUST match training) --------------------------
LOOKBACK, HIDDEN, NUM_LAYERS, DROPOUT, N_FEATURES, BATCH = 168, 256, 3, 0.3, 20, 512

# ---- detection params -------------------------------------------------------
MIN_PEAK_HOURS, MIN_NON_EVENT, ALPHA_FDR, MIN_REL_DROP = 1, 100, 0.05, 0.10
SEED = 42

# ---- segmentation -----------------------------------------------------------
RELIABLE_THR, OCCASIONAL_THR, MIN_EVENTS = 0.60, 0.20, 3
BASE_FEATS = ["mean_kWh", "std_kWh", "cv", "evening_peak_kWh",
              "peak_to_offpeak_ratio", "temp_slope"]

# ---- LLM (local Ollama) -----------------------------------------------------
LLM_MODEL = "qwen2.5"
LLM_HOST  = "http://localhost:11434"


def profile_of(s):
    """Map a signal name to its profile family (A/B/C/P/P0)."""
    s = str(s)
    return "P0" if s.startswith("P0") else ("P" if s.startswith("P") else s.split("_")[0])

"""Run the pipeline in order. Each stage is skipped if its outputs are newer than
its inputs. Use --force to rerun everything.

  python pipeline/run_all.py            # incremental
  python pipeline/run_all.py --force    # full rebuild
"""
import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

PIPE = Path(__file__).resolve().parent
STAGES = [
    ("01_detect.py",   [C.EVENTS],              [C.LSTM_WEIGHTS, C.TSA]),
    ("02_segment.py",  [C.TIERS],               [C.EVENTS, C.EARNINGS]),
    ("03_target.py",   [C.RF_PKL, C.SCORES],    [C.TIERS]),
    ("04_insights.py", [C.SUMMARY, C.CONVERSION], [C.TIERS, C.SCORES, C.EVENTS]),
]


def fresh(outs, ins):
    if not all(Path(o).exists() for o in outs):
        return False
    omin = min(Path(o).stat().st_mtime for o in outs)
    present = [Path(i) for i in ins if Path(i).exists()]
    return bool(present) and omin >= max(p.stat().st_mtime for p in present)


def main():
    force = "--force" in sys.argv
    for name, outs, ins in STAGES:
        if not force and fresh(outs, ins):
            print(f"skip {name} (up to date)")
            continue
        print(f"run  {name}")
        subprocess.run([sys.executable, str(PIPE / name)], check=True)
    print("pipeline complete.")


if __name__ == "__main__":
    main()

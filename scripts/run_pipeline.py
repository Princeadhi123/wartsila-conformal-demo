"""Regenerate all pipeline artifacts: python scripts/run_pipeline.py

Runs the marine track (ensemble + conformal on the UCI naval propulsion
dataset) then the energy track (quantile regression + CQR/ACI on Finnish
residual load). Each takes roughly 1-2 minutes. Artifacts land in
`artifacts/` and are consumed by the demo app and the tests.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uaagent import energy, marine  # noqa: E402


def main() -> None:
    t0 = time.time()
    print("== marine track ==")
    mm = marine.run()
    for tgt, tm in mm["targets"].items():
        s = tm["test"]
        print(
            f"{tgt}: coverage {s['coverage_adaptive']:.3f} adaptive "
            f"({s['coverage_split']:.3f} split), "
            f"false alarms naive {s['naive_false_alarm_rate']:.3f} -> "
            f"ua {s['ua_false_alarm_rate']:.4f}"
        )
    print("== energy track ==")
    em = energy.run()
    c = em["coverage"]
    print(
        f"energy coverage: raw {c['raw_quantile_regression']:.3f} -> "
        f"static CQR {c['cqr_static']:.3f} -> ACI {c['cqr_aci']:.3f} "
        f"(target {1 - em['alpha']:.0%})"
    )
    r = em["reserve"]["conformal_dynamic"]
    print(
        f"dynamic reserve: mean {r['mean_reserve_mw']:.0f} MW, "
        f"shortfall freq {r['shortfall_frequency']:.3f}"
    )
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

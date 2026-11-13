"""
Repeatable HIL test matrix (software HIL, Stage 1).

    python run_hil_matrix.py                    # local results only (hil_results/)
    python run_hil_matrix.py --firebase         # ALSO write fault/sms/run records to Firebase (persistent)

Runs T1..T6 (+ T4A/T4B and the T5 raw-input comparison), evaluates the pass/fail
checks, and writes CSVs, event JSON, plots and test_matrix.md.
Same seeds, same result every time.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from hil.config import HILConfig, EXPERIMENTAL_LABEL
from hil.report import write_outputs
from hil.runner import HILRunner
from hil.scenarios import build_scenarios
from hil.sinks import NullSink

ORDER = ["T1", "T2", "T3", "T4A", "T4B", "T5", "T5R", "T6", "T7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--firebase", action="store_true", help="also write records to Firebase (never deleted by this code)")
    ap.add_argument("--out", default="hil_results")
    ap.add_argument("--profile", choices=["standard", "demo"], default="standard",
                    help="demo = the same tests squeezed into 15-20 simulated minutes")
    args = ap.parse_args()

    cfg = HILConfig()
    sink = NullSink()
    if args.firebase:
        from utils.firebase import db_ref            # noqa: F401  (initialises firebase_admin)
        from firebase_admin import db
        from hil.sinks import FirebaseSink
        sink = FirebaseSink(db)

    scenarios = build_scenarios(args.profile)
    wall = time.time()
    results, table, failed = {}, [], 0
    for i, sid in enumerate(ORDER):
        res = HILRunner(cfg, sink=sink, sms_backend="unavailable", wall_start=wall + i).run(scenarios[sid])
        results[sid] = res
        checks = scenarios[sid].checks(res)
        failed += sum(not ok for _, ok, _ in checks)
        table.append((sid, scenarios[sid], res, checks))

    # cross-run check: simulated mitigation must lower the peak (Run B vs Run A)
    a, b = results["T4A"], results["T4B"]
    cross = [("Run B peak collector temperature lower than Run A",
              b.metrics["peak_collector_c"] < a.metrics["peak_collector_c"] - 10,
              f"A {a.metrics['peak_collector_c']} C vs B {b.metrics['peak_collector_c']} C")]
    v, r = results["T5"], results["T5R"]
    cross.append(("validated inputs used less heater energy than raw inputs after the S3 fault",
                  v.metrics["heater_on_s"] < r.metrics["heater_on_s"],
                  f"validated {v.metrics['heater_on_s']:.0f} s vs raw {r.metrics['heater_on_s']:.0f} s"))
    failed += sum(not ok for _, ok, _ in cross)

    out = Path(args.out)
    write_outputs(results, out, table, cfg)

    L = ["# STMaCS HIL test matrix (software HIL)", "",
         f"All data below is **HIL SIMULATED** (digital twin + injected faults). {EXPERIMENTAL_LABEL}.", "",
         "| Test | Scenario | Events (type / severity / detection delay) | Peak S1 (°C) | Result |", "|---|---|---|---|---|"]
    for sid, sc, res, checks in table:
        evs = "; ".join(f"{e['fault_type']}{'/' + e['subject'] if e['fault_type'] == 'SENSOR_FAULT' else ''} "
                        f"{e['peak_severity']} {e['detection_delay_s']} s" for e in res.events) or "none"
        L.append(f"| {sid} | {sc.name} | {evs} | {res.metrics['peak_collector_c']} | "
                 f"{'PASS' if all(ok for _, ok, _ in checks) else 'FAIL'} |")
    L += ["", "## Checks", ""]
    for sid, sc, res, checks in table:
        L.append(f"### {sid} - {sc.name}")
        L.append(f"*Purpose:* {sc.purpose}  \n*Run id:* `{res.run_id}`")
        L.append("")
        for name, ok, det in checks:
            L.append(f"- {'PASS' if ok else '**FAIL**'} - {name}" + (f" ({det})" if det else ""))
        L.append("")
    L += ["### Cross-run checks", ""]
    for name, ok, det in cross:
        L.append(f"- {'PASS' if ok else '**FAIL**'} - {name} ({det})")
    (out / "test_matrix.md").write_text("\n".join(L) + "\n")

    for sid, sc, res, checks in table:
        bad = [n for n, ok, _ in checks if not ok]
        print(f"{sid:4s} {'PASS' if not bad else 'FAIL ' + str(bad)}")
    for name, ok, det in cross:
        print(f"{'PASS' if ok else 'FAIL'} {name} ({det})")
    print(f"\nresults written to {out}/   firebase: {'yes' if args.firebase else 'no'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

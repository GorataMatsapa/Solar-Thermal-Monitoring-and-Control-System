"""CSV, JSON, markdown and plots for the engineering report."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import config_as_dict, EXPERIMENTAL_LABEL

C = {"collector": "#d9480f", "tank_top": "#1c7ed6", "tank_bot": "#74c0fc", "cmd": "#2b8a3e",
     "act": "#862e9c", "flow": "#0b7285", "warn": "#f08c00", "major": "#e8590c", "crit": "#c92a2a"}
SEV_COL = {"warning": C["warn"], "major": C["major"], "critical": C["crit"]}


def _style(ax):
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _markers(ax, res, with_labels=True):
    for e in res.events:
        ax.axvline(e["detection_s"] / 60, color=SEV_COL[e["peak_severity"]], lw=1, ls="--", alpha=0.8)
    for a in res.actions:
        if a["domain"] == "SIMULATION" and a["action"] == "SIM_SOLAR_INPUT_REDUCED":
            ax.axvline(a["t"] / 60, color="#2b8a3e", lw=1.4, ls="-", alpha=0.9)


def plot_scenario(res, path):
    df, cfg = res.df, res.cfg
    m = df.t_s / 60
    fig, ax = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.2]})
    ax[0].plot(m, df.twin_s1, color=C["collector"], label="S1 collector (twin truth)")
    ax[0].plot(m, df.twin_s3, color=C["tank_top"], label="S3 tank top")
    ax[0].plot(m, df.twin_s4, color=C["tank_bot"], label="S4 tank bottom")
    bad = df[df.rel_s1 + df.rel_s3 + df.rel_s4 < 3]
    if len(bad):
        ax[0].scatter(bad.t_s / 60, bad.meas_s1.where(bad.rel_s1 == 0, bad.meas_s3.where(bad.rel_s3 == 0, bad.meas_s4)).clip(-20, 160),
                      s=5, color="k", alpha=0.5, label="rejected readings (as measured)")
    for lvl, name in zip(cfg.collector_c, ("warning", "major", "critical")):
        ax[0].axhline(lvl, color=SEV_COL[name], lw=0.8, ls=":")
    ax[0].set_ylabel("Temperature (°C)")
    ax[0].legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    ax[1].step(m, df.pump_cmd + 0.04, where="post", color=C["cmd"], label="pump COMMAND")
    ax[1].step(m, df.pump_actual - 0.04, where="post", color=C["act"], label="pump ACTUAL (fan / relay)")
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["OFF", "ON"]); ax[1].set_ylim(-0.3, 1.35)
    ax1b = ax[1].twinx()
    ax1b.plot(m, df.flow_meas_lpm, color=C["flow"], lw=1, alpha=0.8, label="measured flow")
    ax1b.set_ylabel("Flow (L/min)", color=C["flow"]); ax1b.set_ylim(-0.1, 3)
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=3, frameon=False)
    ax[2].fill_between(m, df.max_open_severity, step="post", color="#adb5bd", alpha=0.7)
    ax[2].set_yticks([0, 1, 2, 3]); ax[2].set_yticklabels(["none", "warning", "major", "critical"])
    ax[2].set_ylabel("Open fault"); ax[2].set_xlabel("Simulated time (min)")
    for a in ax:
        _style(a)
    for a in ax[:2]:
        _markers(a, res)
        for inj in res.scenario.injections:
            a.axvspan(inj.start_s / 60, (inj.end_s if inj.end_s else res.scenario.duration_s) / 60, color="#ffe3e3", alpha=0.35, lw=0)
    fig.suptitle(f"{res.scenario.sid} - {res.scenario.name}   [HIL SIMULATED DATA]", fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.005, "pink = injected fault, dashed = fault detected, green line = simulated protection started. "
             + EXPERIMENTAL_LABEL + ".", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_t4_comparison(a, b, path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ax[0].plot(a.df.t_s / 60, a.df.twin_s1, color="#c92a2a", lw=2, label="Run A: no mitigation")
    ax[0].plot(b.df.t_s / 60, b.df.twin_s1, color="#2b8a3e", lw=2, label="Run B: simulated autonomous mitigation")
    for lvl, name in zip(a.cfg.collector_c, ("warning", "major", "critical")):
        ax[0].axhline(lvl, color=SEV_COL[name], lw=0.8, ls=":")
        ax[0].text(0.3, lvl + 0.8, f"{name} {lvl:.0f} °C", fontsize=7, color=SEV_COL[name])
    def first(res, types):
        ev = [e for e in res.events if e["fault_type"] in types]
        return ev[0] if ev else None
    sa, sb = first(a, ("STAGNATION", "COLLECTOR_OVERTEMP")), first(b, ("STAGNATION", "COLLECTOR_OVERTEMP"))
    if sb:
        ax[0].axvline(sb["detection_s"] / 60, color="#1c7ed6", ls="--", lw=1)
        ax[0].text(sb["detection_s"] / 60 + 0.3, 40, "stagnation detected", fontsize=8, color="#1c7ed6", rotation=90)
    mit = next((x for x in b.actions if x["action"] == "SIM_SOLAR_INPUT_REDUCED"), None)
    if mit:
        ax[0].axvline(mit["t"] / 60, color="#2b8a3e", ls="-", lw=1.2)
        ax[0].text(mit["t"] / 60 + 0.3, 40, "simulated mitigation starts", fontsize=8, color="#2b8a3e", rotation=90)
    ax[0].axvline(5, color="#868e96", ls=":", lw=1); ax[0].text(5.3, 128, "pump fails", fontsize=8, color="#868e96")
    ax[0].set_ylabel("Collector temperature S1 (°C)"); ax[0].legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62), fontsize=9)
    ax[1].plot(a.df.t_s / 60, a.df.irradiance_wm2, color="#c92a2a", lw=1.2, label="irradiance, run A")
    ax[1].plot(b.df.t_s / 60, b.df.irradiance_wm2, color="#2b8a3e", lw=1.2, label="irradiance reaching collector, run B (simulated shading)")
    ax[1].set_ylabel("W/m²"); ax[1].set_xlabel("Simulated time (min)"); ax[1].legend(frameon=False, fontsize=8)
    for x in ax:
        _style(x)
    fig.suptitle("Test 4 - stagnation: unmitigated vs simulated mitigation   [HIL SIMULATED DATA]", x=0.01, ha="left")
    fig.text(0.01, 0.005, "Simulation only: shading and the solenoid dump exist only in the digital twin. " + EXPERIMENTAL_LABEL + ".",
             fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.015, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_t5_comparison(v, r, path):
    fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    ax[0].plot(v.df.t_s / 60, v.df.twin_s3, color=C["tank_top"], label="S3 true (twin)")
    ax[0].plot(v.df.t_s / 60, v.df.meas_s3.clip(-30, 120), color="#495057", lw=0.9, alpha=0.8, label="S3 as measured (clipped to -30 °C)")
    ax[0].set_ylabel("S3 (°C)"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].step(v.df.t_s / 60, v.df.heater_cmd, where="post", color="#2b8a3e", label="backup heater, validated inputs")
    ax[1].step(r.df.t_s / 60, r.df.heater_cmd + 0.05, where="post", color="#c92a2a", label="backup heater, raw inputs")
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["OFF", "ON"]); ax[1].legend(frameon=False, fontsize=8, loc="upper left")
    ax[2].step(v.df.t_s / 60, v.df.pump_actual, where="post", color="#2b8a3e", label="pump, validated inputs")
    ax[2].step(r.df.t_s / 60, r.df.pump_actual + 0.05, where="post", color="#c92a2a", label="pump, raw inputs")
    ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["OFF", "ON"]); ax[2].set_xlabel("Simulated time (min)")
    ax[2].legend(frameon=False, fontsize=8, loc="lower left")
    for inj in v.scenario.injections:
        for a in ax:
            a.axvspan(inj.start_s / 60, inj.end_s / 60, color="#ffe3e3", alpha=0.4, lw=0)
        ax[0].text(inj.start_s / 60 + 0.2, ax[0].get_ylim()[1] * 0.9, inj.target, fontsize=9)
    for a in ax:
        _style(a)
    fig.suptitle("Test 5 - sensor failure: validated vs raw controller inputs   [HIL SIMULATED DATA]", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(results: dict, outdir: Path, summary_rows, cfg):
    outdir = Path(outdir)
    (outdir / "csv").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    (outdir / "events").mkdir(exist_ok=True)
    for sid, res in results.items():
        res.df.to_csv(outdir / "csv" / f"{sid}_timeseries.csv", index=False)
        (outdir / "events" / f"{sid}_events.json").write_text(
            json.dumps({"run_id": res.run_id, "events": res.events, "actions": res.actions, "sms": res.sms},
                       indent=2, default=str))
        plot_scenario(res, outdir / "plots" / f"{sid}.png")
    plot_t4_comparison(results["T4A"], results["T4B"], outdir / "plots" / "T4_comparison.png")
    plot_t5_comparison(results["T5"], results["T5R"], outdir / "plots" / "T5_comparison.png")
    (outdir / "thresholds_used.json").write_text(json.dumps(config_as_dict(cfg), indent=2))

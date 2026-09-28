"""Thesis figure: 3-stage maturity LCoE roadmap (V3.25 reference).

How the levelised cost falls as the V3.25 system moves through three
development stages -- Early, Mid, Mature -- driven purely by longer component
lives and less hands-on labour. The kite DESIGN and all material unit prices
are held at today's values; only the lifetimes and the labour effort improve.
So this figure isolates how maturity (reliability + reduced manual effort)
moves LCoE, NOT how the design or material prices would have to change to
reach those lifetimes (out of scope).

This is the roadmap use case, distinct from the TEF design optimisation:
  - the tether bending master curve is turned OFF; the flat, staged
    operational-wear life governs the tether replacement instead (its
    design safety factor therefore does not apply here);
  - the kite canopy life, tether operational life, base-crew operating
    hours and per-flight-hour maintenance step through the three stages.

The staged inputs come from the ``development_stage`` presets in
``ecomo.maturity`` -- the single source of truth also used by run_ecomo -- so
this figure and the config-driven runs agree exactly:
  Stage   canopy life  tether op life  operating h/wk  maint h/flight-h
  Early   100 h        250 h           20             0.25
  Mid     500 h        1000 h          5              0.10
  Mature  5000 h       5000 h          1              0.01

Each stage is applied by setting ``development_stage`` alone; the preset then
supplies the lifetimes and labour hours and puts the tether into
operational-wear mode (bending master curve off), so the staged operational
life governs the replacement. The weekly operating hours are converted to a
per-day value using the wind resource's operating-day count N_op. The
permanent cost-model offsets (sensor 9000, ultracapacitor 30000, labour
50 EUR/h, automation 0) are baked into the example config and apply at every
stage.

Style matches fig_lcoe_improvement_roadmap.py; exports vector PDF + PNG and
prints a companion table of the swept values.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecomo.analysis.sweep import (          # noqa: E402
    DISPLAY_CATEGORIES, SweepRunner, display_breakdown)
from ecomo.analysis.plots import (          # noqa: E402
    _PRETTY, _SUBSYSTEM_COLORS)
from ecomo.maturity import STAGE_PRESETS    # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

SETTINGS = PROJECT_ROOT / "config" / "example" / "economic_settings_V3_example.yml"
OUT_DIR = PROJECT_ROOT / "figures" / "EconomicModel"

# Three maturity stages, low -> high maturity, mapped to their preset keys.
STAGES = ("Early", "Mid", "Mature")
STAGE_KEY = {"Early": "early", "Mid": "mid", "Mature": "mature"}


def main():
    with SweepRunner(SETTINGS) as runner:
        contrib = {c: [] for c in DISPLAY_CATEGORIES}
        totals = []
        table = {}
        for stage in STAGES:
            key = STAGE_KEY[stage]
            # Drive the run through the development_stage preset alone: the
            # single source of truth (ecomo.maturity) supplies every lever
            # and forces the operational-wear tether mode, so this matches
            # the config-driven run_ecomo exactly.
            model, eco = runner.run([("settings", "development_stage", key)])
            met = eco["metrics"]
            crf, aep = met["CRF"], met["AEP"]
            for cat, (cap, op) in display_breakdown(eco).items():
                contrib[cat].append(cap * crf / aep + op / aep)
            totals.append(met["LCoE"])
            preset = STAGE_PRESETS[key]
            table[stage] = {
                "canopy": preset["canopy_lifetime_flight_hours"],
                "tether_op": preset["operational_life_flight_hours"],
                # The per-day operating hours are derived from the weekly
                # target and the wind resource's N_op; read the value the
                # model actually used.
                "hours_per_day": model.inputs.operations.operatingHoursPerDay,
                "maint": preset["maintenance_hours_per_flight_hour"],
                "governing_mode": eco["tether"]["life"]["governing_mode"],
                "lcoe": met["LCoE"],
            }
    contrib = {c: np.array(v) for c, v in contrib.items()}
    totals = np.array(totals)

    # ---- companion table --------------------------------------------------
    print(f"\n{'stage':>7} {'canopy[h]':>10} {'tether_op[h]':>12} "
          f"{'h/day':>6} {'maint[h/fh]':>11} "
          f"{'gov_mode':>11} {'LCoE[EUR/MWh]':>13}")
    for stage in STAGES:
        r = table[stage]
        print(f"{stage:>7} {r['canopy']:>10.0f} {r['tether_op']:>12.0f} "
              f"{r['hours_per_day']:>6.2f} "
              f"{r['maint']:>11.2f} {r['governing_mode']:>11} "
              f"{r['lcoe']:>13.1f}")
    print(f"\nLCoE drop: {totals[0]:.0f} -> {totals[-1]:.0f} EUR/MWh "
          f"({100 * (1 - totals[-1] / totals[0]):.0f}% reduction)")

    # ---- figure -----------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active = [c for c in DISPLAY_CATEGORIES if np.any(contrib[c] > 1e-9)]
    save_bars(active, contrib, totals)


def save_bars(active, contrib, totals):
    """Stacked-bar LCoE decomposition across the three maturity stages."""
    x = np.arange(len(STAGES))
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    bottom = np.zeros(len(STAGES))
    for c in active:
        vals = contrib[c]
        ax.bar(x, vals, 0.55, bottom=bottom, color=_SUBSYSTEM_COLORS[c],
               edgecolor="white", linewidth=0.5, label=_PRETTY[c])
        bottom += vals
    for xi, tot in zip(x, totals):
        ax.text(xi, tot + totals.max() * 0.01, f"{tot:.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(list(STAGES), fontsize=10)
    ax.set_xlabel("Development stage", fontsize=10)
    ax.set_ylabel("LCoE  [EUR/MWh]", fontsize=10)
    ax.set_ylim(0, totals.max() * 1.12)
    ax.set_title("TU Delft V3 LCoE maturity roadmap", fontsize=12)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    # Caption: what is held fixed vs what moves.
    # fig.text(0.5, -0.04,
    #          "Material unit costs and the kite design are held at today's "
    #          "values; only component\nlives and labour effort mature. The "
    #          "figure isolates the maturity effect, not the design changes\n"
    #          "needed to achieve those lives (out of scope).",
    #          ha="center", va="top", fontsize=8, color="#555")

    stem = "lcoe_maturity_roadmap_V3"
    fig.savefig(OUT_DIR / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_DIR / stem}.pdf")


if __name__ == "__main__":
    main()

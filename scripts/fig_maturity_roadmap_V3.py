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

Staged inputs (per the handoff spec):
  Stage   canopy life  tether op life  operating h/wk  maint h/flight-h
  Early   100 h        250 h           20              0.50
  Mid     500 h        1000 h          10              0.20
  Mature  5000 h       5000 h          1               0.05

The base-crew operating hours per day are derived from the h/week targets via
the actual operating-day count N_op of the V3 wind resource
(operating_hours_per_day = h_per_week * 52 / N_op). The permanent cost-model
offsets (sensor 9000, ultracapacitor 30000, labour 50 EUR/h, automation 0)
are already baked into the example config and apply at every stage.

Style matches fig_lcoe_improvement_roadmap.py; exports vector PDF + PNG and
prints a companion table of the swept values.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecomo.ecomo.analysis.sweep import (          # noqa: E402
    DISPLAY_CATEGORIES, SweepRunner, display_breakdown)
from ecomo.ecomo.analysis.plots import (          # noqa: E402
    _PRETTY, _SUBSYSTEM_COLORS)
from ecomo.ecomo.eco_hours import annual_operating_days  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

SETTINGS = PROJECT_ROOT / "config" / "example" / "economic_settings_V3_example.yml"
OUT_DIR = PROJECT_ROOT / "figures" / "EconomicModel"

# Three maturity stages, low -> high maturity.
STAGES = ("Early", "Mid", "Mature")
STAGE_INPUTS = {
    #          canopy_life  tether_op_life  hours_per_week  maint_h_per_flight_h
    "Early":  dict(canopy=100.0,  tether_op=250.0,  hours_per_week=20.0, maint=0.50),
    "Mid":    dict(canopy=500.0,  tether_op=1000.0, hours_per_week=10.0, maint=0.20),
    "Mature": dict(canopy=5000.0, tether_op=5000.0, hours_per_week=1.0,  maint=0.05),
}


def stage_overrides(inputs, n_op):
    """Override list placing every lever at one maturity stage.

    The master curve is switched off so the staged operational-wear life
    governs the tether; the operating hours per day are derived from the
    h/week target and the actual operating-day count N_op.
    """
    hours_per_day = inputs["hours_per_week"] * 52.0 / n_op
    return [
        # Roadmap: master curve OFF -> a1/a2 fallback + staged operational
        # life; the operational-wear mode then governs the tether.
        ("cost", "costs.tether.master_curve", None),
        ("cost", "costs.tether.operational_life_enabled", True),
        ("cost", "costs.tether.operational_life_flight_hours",
         inputs["tether_op"]),
        ("cost", "costs.kite.structure.soft.canopy_lifetime_flight_hours",
         inputs["canopy"]),
        ("settings", "operations.operating_hours_per_day", hours_per_day),
        ("settings", "operations.maintenance_hours_per_flight_hour",
         inputs["maint"]),
    ], hours_per_day


def main():
    with SweepRunner(SETTINGS) as runner:
        # Operating-day count of the V3 wind resource (drives the base-crew
        # hours-per-day conversion). Read from any single run.
        model, _ = runner.run([])
        n_op = annual_operating_days(model.inputs.performance)

        contrib = {c: [] for c in DISPLAY_CATEGORIES}
        totals = []
        table = {}
        for stage in STAGES:
            inputs = STAGE_INPUTS[stage]
            overrides, hours_per_day = stage_overrides(inputs, n_op)
            _, eco = runner.run(overrides)
            met = eco["metrics"]
            crf, aep = met["CRF"], met["AEP"]
            for cat, (cap, op) in display_breakdown(eco).items():
                contrib[cat].append(cap * crf / aep + op / aep)
            totals.append(met["LCoE"])
            table[stage] = {
                **inputs,
                "hours_per_day": hours_per_day,
                "governing_mode": eco["tether"]["life"]["governing_mode"],
                "lcoe": met["LCoE"],
            }
    contrib = {c: np.array(v) for c, v in contrib.items()}
    totals = np.array(totals)

    # ---- companion table --------------------------------------------------
    print(f"\nV3 operating days N_op = {n_op:.1f} d/yr")
    print(f"\n{'stage':>7} {'canopy[h]':>10} {'tether_op[h]':>12} "
          f"{'h/wk':>5} {'h/day':>6} {'maint[h/fh]':>11} "
          f"{'gov_mode':>11} {'LCoE[EUR/MWh]':>13}")
    for stage in STAGES:
        r = table[stage]
        print(f"{stage:>7} {r['canopy']:>10.0f} {r['tether_op']:>12.0f} "
              f"{r['hours_per_week']:>5.0f} {r['hours_per_day']:>6.2f} "
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
    ax.set_xticklabels([f"{s}\n({STAGE_INPUTS[s]['hours_per_week']:.0f} h/wk)"
                        for s in STAGES], fontsize=10)
    ax.set_ylabel("LCoE  [EUR/MWh]", fontsize=10)
    ax.set_ylim(0, totals.max() * 1.12)
    ax.set_title("V3.25 LCoE maturity roadmap", fontsize=12)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    # Caption: what is held fixed vs what moves.
    fig.text(0.5, -0.04,
             "Material unit costs and the kite design are held at today's "
             "values; only component\nlives and labour effort mature. The "
             "figure isolates the maturity effect, not the design changes\n"
             "needed to achieve those lives (out of scope).",
             ha="center", va="top", fontsize=8, color="#555")

    stem = "lcoe_maturity_roadmap_V3"
    fig.savefig(OUT_DIR / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_DIR / stem}.pdf")


if __name__ == "__main__":
    main()

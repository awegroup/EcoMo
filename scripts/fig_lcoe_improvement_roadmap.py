"""Thesis figure: LCoE improvement roadmap (V3 reference).

How the levelised cost falls as three development levers move from today's
pessimistic values (f = 0) to the target optimistic ones (f = 1), decomposed
by cost category. The three levers hit disjoint subsystems and none touches
AEP (availability fixed at 0.90), so the total is the superposition of the
three individual 1-D responses evaluated at the same improvement factor f:

  LCoE(f) = LCoE_fixed + [C_kite(f) + C_tether(f) + C_crew(f)] / AEP

Levers, linearly interpolated in their physical inputs over f in {0..1}:
  - Kite (canopy) life        : 250 -> 5000 flight-h
  - Tether operational life   : 250 -> bending life (~13.9 kh at 6 mm, a=0.90);
                                operational mode stays ON and smoothly meets
                                the bending limit at f = 1 (no mode switch)
  - Ground crew               : automation 0 -> 1 (operation labour ~ (1-f));
                                maintenance 318 -> 52 h/yr (1 h/day -> 1 h/week)

Because replacement cost scales as 1/life while the inputs move linearly in
life, the LCoE drop is front-loaded (steep early, flat late) -- diminishing
returns, kept honest by interpolating in life (not in 1/life).

Everything else is held at the baseline (availability 0.90, financials, D/d,
drum, storage, tether CAPEX at the current 6 mm value, canopy load exponent
m = 0). Display-level cost regroup: Ground crew is its own band.

Style matches massmodel/thesis_cost_figure.py; exports vector PDF + a
companion table of the swept values.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
OUT_FILE = "lcoe_improvement_roadmap_V3.pdf"

# Fine grid for the smooth stacked-area curves; the coarse table points
# {0, 0.2, ..., 1.0} land exactly on it (step 0.02).
F_GRID = np.linspace(0.0, 1.0, 51)
F_TABLE = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Lever endpoints
KITE_LIFE = (250.0, 5000.0)          # flight-h
TETHER_LIFE_LO = 250.0               # flight-h; hi = actual bending life (computed)
MAINT_HR_PER_YR = (None, 52.0)       # lo computed from 1 h/day * N_op; hi = 1 h/week


def scenario_overrides(f, bending_life, n_op):
    """Override list placing all three levers at improvement factor f."""
    kite_life = KITE_LIFE[0] + f * (KITE_LIFE[1] - KITE_LIFE[0])
    tether_life = TETHER_LIFE_LO + f * (bending_life - TETHER_LIFE_LO)
    maint_lo = 1.0 * n_op                       # 1 h/day * operating days
    maint_hr_yr = maint_lo + f * (MAINT_HR_PER_YR[1] - maint_lo)
    maint_h_per_day = maint_hr_yr / n_op        # model input is per operating day
    return [
        ("cost", "costs.kite.structure.soft.canopy_lifetime_flight_hours",
         kite_life),
        ("cost", "costs.tether.operational_life_enabled", True),
        ("cost", "costs.tether.operational_life_flight_hours", tether_life),
        ("settings", "operations.automation", f),
        ("settings", "operations.maintenance_hours_per_day", maint_h_per_day),
    ], dict(kite_life=kite_life, tether_life=tether_life,
            automation=f, maint_hr_yr=maint_hr_yr)


def main():
    with SweepRunner(SETTINGS) as runner:
        # Actual bending life at the baseline (operational off -> bending
        # governs); also gives the model to read N_op from.
        model, eco0 = runner.run([
            ("cost", "costs.tether.operational_life_enabled", True),
            ("cost", "costs.tether.operational_life_flight_hours", 1e9)])
        bending_life = eco0["tether"]["life"]["bending_flight_hours"]
        n_op = annual_operating_days(model.inputs.performance)

        # contribution[category] = array over the fine grid [EUR/MWh]
        contrib = {c: [] for c in DISPLAY_CATEGORIES}
        totals = []
        table = {}
        for f in F_GRID:
            overrides, info = scenario_overrides(f, bending_life, n_op)
            _, eco = runner.run(overrides)
            met = eco["metrics"]
            crf, aep = met["CRF"], met["AEP"]
            for cat, (cap, op) in display_breakdown(eco).items():
                contrib[cat].append(cap * crf / aep + op / aep)
            totals.append(met["LCoE"])
            if np.any(np.isclose(f, F_TABLE)):
                table[round(float(f), 2)] = {**info, "lcoe": met["LCoE"]}
    contrib = {c: np.array(v) for c, v in contrib.items()}
    totals = np.array(totals)

    # ---- companion table --------------------------------------------------
    print(f"\nbending life = {bending_life:.0f} flight-h ; N_op = {n_op:.1f} d/yr")
    print(f"\n{'f':>4} {'kite_life[h]':>12} {'tether_life[h]':>14} "
          f"{'auto':>5} {'maint[h/yr]':>11} {'LCoE[EUR/MWh]':>13}")
    for f in F_TABLE:
        r = table[round(float(f), 2)]
        print(f"{f:>4.1f} {r['kite_life']:>12.0f} {r['tether_life']:>14.0f} "
              f"{r['automation']:>5.2f} {r['maint_hr_yr']:>11.0f} {r['lcoe']:>13.1f}")
    print(f"\nLCoE drop: {totals[0]:.0f} -> {totals[-1]:.0f} EUR/MWh "
          f"({100*(1-totals[-1]/totals[0]):.0f}% reduction)")

    # ---- both figures share the same computed data -----------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active = [c for c in DISPLAY_CATEGORIES if np.any(contrib[c] > 1e-9)]
    save_area(active, contrib, totals)
    save_bars(active, contrib, totals)


def _finish_axes(ax, ymax):
    ax.set_xlabel("Improvement factor "f"f (0 = today, 1 = target)",
                  fontsize=10)
    ax.set_ylabel("LCoE  [EUR/MWh]", fontsize=10)
    ax.set_xticks(F_TABLE)
    ax.set_ylim(0, ymax * 1.10)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9, ncol=2)


def _save(fig, stem):
    fig.savefig(OUT_DIR / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_DIR / stem}.pdf")


def save_area(active, contrib, totals):
    """Smooth stacked-area version over the fine f grid."""
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.stackplot(F_GRID, *[contrib[c] for c in active],
                 labels=[_PRETTY[c] for c in active],
                 colors=[_SUBSYSTEM_COLORS[c] for c in active],
                 alpha=0.9, edgecolor="none")
    ax.plot(F_GRID, totals, color="#222", lw=1.6)             # total LCoE
    for f, tot in ((0.0, totals[0]), (1.0, totals[-1])):
        ax.annotate(f"{tot:.0f}", xy=(f, tot),
                    xytext=(3 if f == 0 else -3, 10),
                    textcoords="offset points",
                    ha="left" if f == 0 else "right", fontsize=9,
                    color="#222", fontweight="bold")
    ax.set_xlim(0, 1)
    _finish_axes(ax, totals.max())
    _save(fig, "lcoe_improvement_roadmap_V3_area")


def save_bars(active, contrib, totals):
    """Discrete stacked-bar version at the six coarse f points."""
    idx = [int(round(f * (len(F_GRID) - 1))) for f in F_TABLE]  # grid indices
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    bottom = np.zeros(len(F_TABLE))
    for c in active:
        vals = contrib[c][idx]
        ax.bar(F_TABLE, vals, 0.13, bottom=bottom, color=_SUBSYSTEM_COLORS[c],
               edgecolor="white", linewidth=0.5, label=_PRETTY[c])
        bottom += vals
    for f, tot in zip(F_TABLE, totals[idx]):
        ax.text(f, tot + 8, f"{tot:.0f}", ha="center", va="bottom",
                fontsize=8.5, color="#333")
    _finish_axes(ax, totals.max())
    _save(fig, "lcoe_improvement_roadmap_V3_bars")


if __name__ == "__main__":
    main()

"""Thesis figure: per-component replacement-cost composition (V3 reference).

Two panels, reference-relative (no absolute euros -- unit prices are
confidential):
  (a) Share [%] of the total annual replacement stream per component,
      horizontal bars sorted descending, colour-grouped by wear driver.
  (b) Effective replacement interval [yr] = 1 / f_repl for the same
      components (same order/colours) -- the physical driver of (a).

The replacement stream is the sum of the per-component OPEX = f_repl *
CAPEX leaves from the model results; operating labour and the per-kW site
overhead are excluded (they belong to the O&M subsection). Reads the
current baseline results (availability = 0.90, canopy 5000 h).

Style matches massmodel/thesis_cost_figure.py; exports vector PDF.
"""

from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results_example" / "ecomo_results.yml"
OUT_DIR = PROJECT_ROOT / "figures" / "EconomicModel"
OUT_FILE = "replacement_shares_V3.pdf"

# Colour groups by wear driver (thesis palette + one teal for storage)
C_CONSUMABLE = "#E05C2A"   # usage-based flying consumables (kite, tether)
C_GS_HARDWARE = "#3A7DC9"  # calendar-life ground-station hardware
C_STORAGE = "#2CA089"      # cycle-driven storage
C_AVIONICS = "#E8A020"     # calendar-life airborne electronics (KCU)

# Component -> (results-tree OPEX key(s), CAPEX key(s), colour group).
# The airborne sensor is folded into KCU/avionics (one 'KCU/avionics' item,
# as in the component list). Each entry may carry >1 key that is summed.
COMPONENTS = [
    ("Kite canopy",       ["kite.structure"],                C_CONSUMABLE),
    ("Tether",            ["tether"],                        C_CONSUMABLE),
    ("KCU/avionics",      ["kite.avionics", "kite.sensor"],  C_AVIONICS),
    ("Ultracapacitor",    ["gStation.ultracap"],             C_STORAGE),
    ("Winch",             ["gStation.winch"],                C_GS_HARDWARE),
    ("Gearbox",           ["gStation.gearbox"],              C_GS_HARDWARE),
    ("Generator",         ["gStation.gen"],                  C_GS_HARDWARE),
    ("Power converter",   ["gStation.powerConv"],            C_GS_HARDWARE),
    ("Launch & recovery", ["gStation.lls"],                  C_GS_HARDWARE),
]

GROUP_LABELS = [
    (C_CONSUMABLE,  "Usage-based consumables (kite, tether)"),
    (C_AVIONICS,    "Airborne electronics (KCU), calendar life"),
    (C_GS_HARDWARE, "Ground-station hardware, calendar life"),
    (C_STORAGE,     "Cycle-driven storage"),
]

PROJECT_LIFE_YR = 25.0


def load_components():
    """Return list of (name, opex, capex, colour), replacement stream only."""
    data = yaml.safe_load(RESULTS.read_text(encoding="utf-8"))
    opex = data["cost_breakdown"]["opex_eur_per_year"]
    capex = data["cost_breakdown"]["capex_eur"]
    rows = []
    for name, keys, colour in COMPONENTS:
        o = sum(opex.get(k, 0.0) for k in keys)
        c = sum(capex.get(k, 0.0) for k in keys)
        if o <= 0:
            raise ValueError(
                f"Component '{name}' has zero replacement OPEX -- it would "
                f"vanish from panel (a). Check that its replacement term is "
                f"charged in the model.")
        rows.append((name, o, c, colour))
    return rows


def main():
    rows = load_components()
    rows.sort(key=lambda r: r[1], reverse=True)         # descending by OPEX
    names = [r[0] for r in rows]
    opex = np.array([r[1] for r in rows])
    capex = np.array([r[2] for r in rows])
    colours = [r[3] for r in rows]

    shares = 100.0 * opex / opex.sum()
    interval = capex / opex                              # 1 / f_repl [yr]
    y = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   constrained_layout=True)

    # -- panel (a): share of annual replacement cost -----------------------
    ax1.barh(y, shares, color=colours, height=0.65,
             edgecolor="white", linewidth=0.4)
    for i, s in enumerate(shares):
        ax1.text(s + 0.6, i, f"{s:.1f}%", va="center", ha="left",
                 fontsize=8.5, color="#444")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()                                  # largest at top
    ax1.set_xlabel("Share of annual replacement cost [%]", fontsize=10)
    ax1.set_xlim(0, shares.max() * 1.18)
    ax1.set_title("(a)", fontsize=11, loc="left", pad=8)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(left=False)
    ax1.grid(axis="x", alpha=0.2)

    # -- panel (b): effective replacement interval -------------------------
    ax2.barh(y, interval, color=colours, height=0.65,
             edgecolor="white", linewidth=0.4)
    for i, iv in enumerate(interval):
        ax2.text(iv + 0.5, i, f"{iv:.1f} yr", va="center", ha="left",
                 fontsize=8.5, color="#444")
    ax2.axvline(PROJECT_LIFE_YR, color="#888", ls="--", lw=1.0)
    ax2.text(PROJECT_LIFE_YR, len(rows) - 0.4, " project life",
             fontsize=8, color="#888", va="bottom", ha="left", rotation=90)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])                             # shared order with (a)
    ax2.invert_yaxis()
    ax2.set_xlabel("Replacement interval [yr]", fontsize=10)
    ax2.set_xlim(0, max(interval.max(), PROJECT_LIFE_YR) * 1.15)
    ax2.set_title("(b)", fontsize=11, loc="left", pad=8)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(left=False)
    ax2.grid(axis="x", alpha=0.2)

    # -- shared colour-group legend ----------------------------------------
    handles = [mpatches.Patch(color=c, label=lbl) for c, lbl in GROUP_LABELS]
    ax1.legend(handles=handles, fontsize=8, loc="lower right",
               framealpha=0.9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / OUT_FILE, format="pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / OUT_FILE.replace(".pdf", ".png"), dpi=150,
                bbox_inches="tight")
    print(f"Saved to {OUT_DIR / OUT_FILE}")
    print(f"Total replacement stream = {opex.sum():.0f} EUR/yr (not plotted)")


if __name__ == "__main__":
    main()

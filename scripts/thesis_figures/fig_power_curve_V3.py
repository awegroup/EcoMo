"""Thesis figure: mean electrical power curve of the fixed V3 design.
 
Reads the AWESPA power-curve output (data/power_curves.yml) and plots the
mean electrical cycle power against the reference wind speed at 200 m.
Writes figures/EconomicModel/power_curve_V3.{pdf,png}.
 
Run from the ecomo repo root:  python scripts/thesis_figures/fig_power_curve_V3.py
"""
 
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
 
ROOT = Path(__file__).resolve().parents[2]          # ecomo repo root
DATA = ROOT / "data" / "power_curves.yml"
OUT = ROOT / "figures" / "EconomicModel"
 
plt.rcParams.update({"font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150})
 
 
def main():
    pc = yaml.safe_load(open(DATA))
    mc = pc["metadata"]["model_config"]
 
    ws, pe = [], []
    for e in pc["power_curves"][0]["wind_speed_data"]:
        ws.append(e["wind_speed"])
        pe.append(e["performance"]["electrical_power"]["average_cycle_power"] / 1000.0)
    ws, pe = np.array(ws), np.array(pe)
 
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ws, pe, "-o", color="#1f3b73", ms=4, lw=1.6)
    ax.axhline(mc["nominal_electrical_power"] / 1000.0, ls="--", color="grey", lw=1)
    ax.axvline(mc["cut_in_wind_speed"], ls=":", color="grey", lw=1)
    ax.axvline(mc["cut_out_wind_speed"], ls=":", color="grey", lw=1)
    ax.set_xlabel(r"Reference wind speed at 200 m  [m s$^{-1}$]")
    ax.set_ylabel("Mean electrical power  [kW]")
    ax.set_xlim(0, 26)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
 
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"power_curve_V3.{ext}", bbox_inches="tight")
    print(f"nominal electrical power = {mc['nominal_electrical_power'] / 1000.0:.2f} kW, "
          f"cut-in = {mc['cut_in_wind_speed']}, cut-out = {mc['cut_out_wind_speed']} m/s")
 
 
if __name__ == "__main__":
    main()
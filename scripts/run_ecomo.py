"""Run the ECOMo economic model for the TU Delft V3.25 example case.

Uses the V3.25 soft kite together with the V3 AWESPA power model outputs
(``data/aep_results.yml`` + ``data/power_curves.yml``) to get a first LCoE
insight. The example settings file points its ``input_files.aep_results`` and
``input_files.power_curves`` fields at the AWESPA outputs, which activates
AWESPA-connected mode.

Usage:
    python scripts/run_ecomo.py
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecomo import EcoMo
from ecomo.analysis import maybe_run_analysis


def main():
    """Run the ECOMo V3.25 example case and export results."""
    # ---- paths ------------------------------------------------------------
    configDir = PROJECT_ROOT / "config" / "example"
    settingsPath = configDir / "economic_settings_V3_example.yml"

    resultsDir = PROJECT_ROOT / "results_example"
    resultsDir.mkdir(parents=True, exist_ok=True)
    outputPath = resultsDir / "ecomo_results.yml"

    # ---- initialise and load model -----------------------------------------
    model = EcoMo()
    model.load_configuration(economic_settings_path=settingsPath)

    # ---- compute economics --------------------------------------------------
    results = model.compute_economics(
        output_path=outputPath,
        verbose=True,
        showplot=True,
    )

    # ---- optional analysis layer (only runs if analysis.enabled in YAML) ---
    produced = maybe_run_analysis(
        settingsPath, resultsDir / "analysis", model=model, eco=results)
    if produced:
        print(f"\nAnalysis figures written to {resultsDir / 'analysis'}/")

    # ---- summary -----------------------------------------------------------
    metrics = results["metrics"]
    print("\n" + "=" * 60)
    print("ECONOMIC ANALYSIS COMPLETE (V3.25 EXAMPLE, AWESPA-CONNECTED)")
    print("=" * 60)
    print(f"  LCoE : {metrics['LCoE']:.2f} EUR/MWh")
    print(f"  AEP  : {metrics['AEP']:.3f} MWh/yr")
    print(f"  ICC  : {metrics['ICC']:,.0f} EUR")
    print(f"\n  Output: {outputPath}")
    print("\nComplete ")


if __name__ == "__main__":
    main()

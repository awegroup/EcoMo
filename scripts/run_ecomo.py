"""Run the ECOMo economic model in standalone mode (manual system inputs).

Usage:
    python scripts/run_ecomo.py
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecomo import EcoMoEconomicModel


def main():
    """Run the ECOMo model in standalone mode and export results."""
    # ---- paths ------------------------------------------------------------
    configDir = PROJECT_ROOT / "config" / "test" / "gg_fixed"
    settingsPath = configDir / "economic_settings_GG_fixed.yml"

    resultsDir = PROJECT_ROOT / "results" / "example"
    resultsDir.mkdir(parents=True, exist_ok=True)
    outputPath = resultsDir / "ecomo_results.yml"

    # ---- initialise and load model -----------------------------------------
    model = EcoMoEconomicModel()
    model.load_configuration(economic_settings_path=settingsPath)

    # ---- compute economics --------------------------------------------------
    results = model.compute_economics(
        output_path=outputPath,
        verbose=True,
        showplot=True,
    )

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("ECONOMIC ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n  Output: {outputPath}")
    print("\nAll done!")


if __name__ == "__main__":
    main()

"""Run the ECOMo economic model using AWESPA power model outputs.

The settings file ``economic_settings_GG_fixed_awespa.yml`` points its
``input_files.aep_results`` and ``input_files.power_curves`` fields at
the AWESPA output files, which activates AWESPA-connected mode.

Usage:
    python scripts/run_ecomo_from_awespa.py
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecomo import EcoMoEconomicModel


def main():
    """Run the ECOMo model in AWESPA-connected mode and export results."""
    # ---- paths ------------------------------------------------------------
    configDir = PROJECT_ROOT / "config" / "test" / "gg_fixed"
    settingsPath = configDir / "economic_settings_GG_fixed_awespa.yml"

    resultsDir = PROJECT_ROOT / "results" / "example.awespa"
    resultsDir.mkdir(parents=True, exist_ok=True)
    outputPath = resultsDir / "ecomo_results_awespa.yml"

    # ---- initialise and load model -----------------------------------------
    model = EcoMoEconomicModel()
    model.load_configuration(economic_settings_path=settingsPath)

    # ---- compute economics --------------------------------------------------
    results = model.compute_economics(
        output_path=outputPath,
        verbose=True,
    )

    # ---- summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("ECONOMIC ANALYSIS COMPLETE (AWESPA-CONNECTED MODE)")
    print("=" * 60)
    print(f"\n  Output: {outputPath}")
    print("\nAll done!")


if __name__ == "__main__":
    main()

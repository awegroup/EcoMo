"""Entry point script for ECOMo simulation.

This script demonstrates how to run the ECOMo economic model with
example input data.
"""

import os
import sys

from ecomo.config import configure
from ecomo.eco_system_inputs import eco_system_inputs_example
from ecomo.eco_main import eco_main
from ecomo.eco_display_results import eco_display_results


def main():
    """Run ECOMo simulation with example inputs."""
    print("\nRunning ECOMo Economic Model...\n")

    # ── Configuration ─────────────────────────────────────────────────
    # Change these settings to switch between system configurations.
    #   power : 'GG' (ground-gen) or 'FG' (fly-gen)
    #   wing  : 'fixed' or 'soft'
    # The input files must match the chosen power/wing combination.
    configure(
        name='example_system',
        input_cost_file='eco_cost_inputs_GG_fixed',
        input_model_file='eco_system_inputs_GG_fixed_example',
        power='GG',
        wing='fixed',
    )
    # ─────────────────────────────────────────────────────────────────

    # Import inputs
    print("Loading input parameters...")
    inp = eco_system_inputs_example()

    # Run EcoModel by parsing the inputs
    print("Running simulation...")
    inp, par, eco = eco_main(inp)

    # Display results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    eco_display_results(inp, eco, show=True, save_dir=results_dir)

    print("Outputs saved to results/ directory")


if __name__ == '__main__':
    main()

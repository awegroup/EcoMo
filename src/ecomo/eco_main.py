"""Main orchestration function for ECOMo simulation.

This module contains the main function that orchestrates the entire ECOMo
simulation by importing input data, defining cost model parameters, executing
subsystem modules, and computing relevant metrics.
"""

import os
import pickle
from typing import Dict, Any, Tuple
from .config import eco_settings
from .eco_import_cost_par import eco_import_cost_par
from .eco_kite import eco_kite
from .eco_tether import eco_tether
from .eco_gstation import eco_gstation
from .eco_bos import eco_bos
from .eco_bop import eco_bop
from .eco_metrics import eco_compute_metrics


def eco_main(inp: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Main function to run ECOMo simulation.

    This function orchestrates the ECOMo simulation by importing input data,
    defining cost model parameters, executing subsystem modules (kite, tether,
    ground station, BoS, BoP), and computing relevant metrics.

    Args:
        inp: Dictionary containing input parameters for the ECOMo simulation.

    Returns:
        Tuple containing:
            - inp: Updated input structure after processing.
            - par: Structure containing cost model parameters.
            - eco: Structure containing results and metrics of the ECOMo simulation.
    """
    # Import cost model parameters
    par = eco_import_cost_par()

    # Initialize structure to store results
    eco = {}

    # Kite
    inp, par, eco = eco_kite(inp, par, eco)

    # Tether
    inp, par, eco = eco_tether(inp, par, eco)

    # Ground station
    inp, par, eco = eco_gstation(inp, par, eco)

    # BoS (Balance of System)
    inp, par, eco = eco_bos(inp, par, eco)

    # BoP (Balance of Plant)
    inp, par, eco = eco_bop(inp, par, eco)

    # Compute metrics
    inp, par, eco = eco_compute_metrics(inp, par, eco)

    # Save outputs
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'results'
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save with pickle (Python equivalent of MATLAB .mat files)
    name = eco_settings.name
    with open(os.path.join(output_dir, f'{name}_inp.pkl'), 'wb') as f:
        pickle.dump(inp, f)
    with open(os.path.join(output_dir, f'{name}_par.pkl'), 'wb') as f:
        pickle.dump(par, f)
    with open(os.path.join(output_dir, f'{name}_eco.pkl'), 'wb') as f:
        pickle.dump(eco, f)

    return inp, par, eco

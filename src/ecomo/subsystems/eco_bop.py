"""Balance of Plant (BoP) subsystem economic calculations.

This module computes the capital and operational expenditures for the
BoP subsystem, including array cables, substations, and grid
integration. None of these are modelled yet.
"""

from typing import Dict, Any


def eco_bop() -> Dict[str, Any]:
    """Calculate costs related to the Balance of Plant subsystem.

    Returns:
        dict: The ``eco['BoP']`` results subtree.
    """
    return {
        'arrayCables': {'CAPEX': 0, 'OPEX': 0},
        'substations': {'CAPEX': 0, 'OPEX': 0},
        'gridInt': {'CAPEX': 0, 'OPEX': 0},
    }

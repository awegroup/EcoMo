"""Balance of Plant (BoP) subsystem economic calculations.

This module computes the capital and operational expenditures for the BoP
subsystem, including array cables, substations, and grid integration.
"""

from typing import Dict, Any, Tuple


def eco_bop(inp: Dict[str, Any], par: Dict[str, Any], eco: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Calculate costs related to Balance of Plant subsystem.

    Args:
        inp: Dictionary containing input parameters.
        par: Dictionary containing cost model parameters.
        eco: Dictionary containing economic results.

    Returns:
        Tuple of (inp, par, eco) dictionaries updated with BoP calculations.
    """
    # Initialize BoP structure in eco if not present
    if 'BoP' not in eco:
        eco['BoP'] = {}

    # Array cables
    eco['BoP']['arrayCables'] = {
        'CAPEX': 0,
        'OPEX': 0
    }

    # Substations
    eco['BoP']['substations'] = {
        'CAPEX': 0,
        'OPEX': 0
    }

    # Grid integration
    eco['BoP']['gridInt'] = {
        'CAPEX': 0,
        'OPEX': 0
    }

    return inp, par, eco

"""Balance of System (BoS) subsystem economic calculations.

This module computes the capital and operational expenditures for the BoS
subsystem, including site preparation, foundation, installation, O&M,
and decommissioning.
"""

from typing import Dict, Any, Tuple


def eco_bos(inp: Dict[str, Any], par: Dict[str, Any], eco: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Calculate costs related to Balance of System subsystem.

    Args:
        inp: Dictionary containing input parameters.
        par: Dictionary containing cost model parameters.
        eco: Dictionary containing economic results.

    Returns:
        Tuple of (inp, par, eco) dictionaries updated with BoS calculations.
    """
    # Initialize BoS structure in eco if not present
    if 'BoS' not in eco:
        eco['BoS'] = {}

    # Site preparation
    eco['BoS']['sitePrep'] = {
        'CAPEX': par['BoS']['sitePrep']['p'] * inp['system']['P_e_rated']/1e3
    }

    # Foundation
    eco['BoS']['found'] = {
        'CAPEX': par['BoS']['found']['p'] * inp['system']['P_e_rated']/1e3
    }

    # Installation
    eco['BoS']['install'] = {
        'CAPEX': par['BoS']['install']['p'] * inp['system']['P_e_rated']/1e3
    }

    # Operations & Maintenance
    eco['BoS']['OM'] = {
        'OPEX': par['BoS']['OM']['p'] * inp['system']['P_e_rated']/1e3
    }

    # Decommissioning
    eco['BoS']['decomm'] = {
        'CAPEX': par['BoS']['decomm']['f'] * eco['BoS']['install']['CAPEX']
    }

    return inp, par, eco

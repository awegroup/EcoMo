"""Tether subsystem economic calculations.

This module computes the cost and operational parameters related to the tether,
including cross-sectional area, CAPEX, OPEX, and life estimation.
"""

import numpy as np
from typing import Dict, Any, Tuple
from .config import eco_settings


def eco_tether(inp: Dict[str, Any], par: Dict[str, Any], eco: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Calculate cost and operational parameters related to the tether.

    Args:
        inp: Dictionary containing input parameters.
        par: Dictionary containing cost model parameters.
        eco: Dictionary containing economic results.

    Returns:
        Tuple of (inp, par, eco) dictionaries updated with tether calculations.
    """
    # Initialize tether structure in eco if not present
    if 'tether' not in eco:
        eco['tether'] = {}

    # Tether cross sectional area
    if 'd' in inp['tether']:
        inp['tether']['A'] = np.pi/4 * inp['tether']['d']**2
    elif 'A' in inp['tether']:
        inp['tether']['d'] = np.sqrt(4 * inp['tether']['A'] / np.pi)

    # For brevity in code
    t = inp['tether']

    # CAPEX
    if eco_settings.power == 'GG':
        eco['tether']['CAPEX'] = (par['tether']['p'] * t['A'] *
                                  par['tether']['f_At'] * t['L'] * t['rho'] *
                                  (1 + par['tether']['f_coat']))
    elif eco_settings.power == 'FG':
        eco['tether']['CAPEX'] = (par['tether']['f_mt'] *
                                  (par['tether']['p'] * t['A'] *
                                   par['tether']['f_At'] * t['L'] * t['rho'] *
                                   (1 + par['tether']['f_coat'])))

    # Tether stress
    eco['tether']['sigma'] = np.minimum(
        inp['system']['F_t'] / (par['tether']['f_At'] * t['A']),
        par['tether']['sigma_max']
    )

    # OPEX
    # Tether life estimation due to bending - Relevant for GG
    if eco_settings.power == 'GG':
        exp = (par['tether']['a_1b'] -
               par['tether']['a_2b'] * eco['tether']['sigma']/1e9)
        Nb = 10**exp
        integral_term = (inp['atm']['gw'] /
                        (inp['system']['Dt_cycle']/8760/3600 * Nb))
        integral_term[np.isinf(integral_term)] = 0
        L_bend = 1 / (par['tether']['N_bends'] *
                     np.trapezoid(integral_term, inp['atm']['wind_range']))
        eco['tether']['f_repl_bend'] = 1/L_bend/3  # 3 times correction factor

    # Tether life estimation due to creep - Relevant for FG
    L_creep_coef = par['tether']['L_creep']

    if np.isscalar(L_creep_coef) and not isinstance(L_creep_coef, np.ndarray):
        # If scalar, treat as a constant life
        L_creep = L_creep_coef * np.ones_like(eco['tether']['sigma'])
    else:
        # If array, use polyval
        exp = np.polyval(L_creep_coef, eco['tether']['sigma'] / 1e9)
        L_creep = 10 ** exp
    life_creep = 1 / np.trapezoid(inp['atm']['gw'] / L_creep,
                                   inp['atm']['wind_range'])
    eco['tether']['f_repl_creep'] = 1/life_creep

    # Set tether replacement frequency
    if t['f_repl'] < 0:
        if eco_settings.power == 'GG':
            eco['tether']['f_repl'] = max(eco['tether']['f_repl_bend'],
                                          eco['tether']['f_repl_creep'])
        elif eco_settings.power == 'FG':
            eco['tether']['f_repl'] = eco['tether']['f_repl_creep']
    else:
        eco['tether']['f_repl'] = t['f_repl']

    # Set tether life to infinite if tether life > AWE operational years
    if 1/eco['tether']['f_repl'] > inp['business']['N_y']:
        eco['tether']['f_repl'] = 0

    eco['tether']['OPEX'] = eco['tether']['f_repl'] * eco['tether']['CAPEX']

    return inp, par, eco

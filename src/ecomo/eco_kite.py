"""Kite subsystem economic calculations.

This module computes the capital expenditure (CAPEX) and operational
expenditure (OPEX) associated with the kite subsystem, including structure,
onboard generators, onboard batteries, and avionics.
"""

import numpy as np
from typing import Dict, Any, Tuple
from .config import eco_settings


def eco_kite(inp: Dict[str, Any], par: Dict[str, Any], eco: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Calculate costs related to kite subsystem.

    Args:
        inp: Dictionary containing input parameters.
        par: Dictionary containing cost model parameters.
        eco: Dictionary containing economic results.

    Returns:
        Tuple of (inp, par, eco) dictionaries updated with kite calculations.
    """
    # Initialize kite structure in eco if not present
    if 'kite' not in eco:
        eco['kite'] = {}

    # Structure
    # Find wing area if wingspan and aspect ratio are provided
    if 'b' in inp['kite']['structure']:
        inp['kite']['structure']['A'] = (inp['kite']['structure']['b']**2 /
                                         inp['kite']['structure']['AR'])

    if eco_settings.wing == 'fixed':
        # Fixed wing
        approach = par['kite']['structure']['fixed']['approach']

        if approach == 1:
            # CAPEX
            eco['kite']['structure'] = {
                'CAPEX': (par['kite']['structure']['fixed']['one']['p_str'] *
                         inp['kite']['structure']['m'] +
                         par['kite']['structure']['fixed']['one']['p_wet'] *
                         inp['kite']['structure']['A'])
            }
        elif approach == 2:
            # CAPEX
            eco['kite']['structure'] = {
                'CAPEX': ((1 + par['kite']['structure']['fixed']['two']['f_man']) *
                         (par['kite']['structure']['fixed']['two']['p_uni'] *
                          inp['kite']['structure']['m_uni'] +
                          par['kite']['structure']['fixed']['two']['p_triax'] *
                          inp['kite']['structure']['m_tri']))
            }

        # OPEX
        eco['kite']['structure']['OPEX'] = (inp['kite']['structure']['f_repl'] *
                                            eco['kite']['structure']['CAPEX'])

    elif eco_settings.wing == 'soft':
        # Soft wing
        # CAPEX
        eco['kite']['structure'] = {
            'CAPEX': ((par['kite']['structure']['soft']['p_fabric'] +
                      par['kite']['structure']['soft']['p_bridle']) *
                     inp['kite']['structure']['A'])
        }

        # OPEX
        if inp['kite']['structure']['f_repl'] < 0:
            # Calculate load factor
            LF = np.trapezoid(inp['atm']['gw'] * inp['system']['F_t'] /
                              np.max(inp['system']['F_t']),
                              inp['atm']['wind_range'])
            inp['kite']['structure']['f_repl'] = LF / par['kite']['structure']['soft']['L_str']

        eco['kite']['structure']['OPEX'] = (inp['kite']['structure']['f_repl'] *
                                            eco['kite']['structure']['CAPEX'])

    # Onboard generators and batteries
    if eco_settings.power == 'FG':
        # Fly-gen: Onboard generators
        eco['kite']['obGen'] = {
            'CAPEX': par['kite']['obGen']['p'] * inp['system']['P_e_rated']/1e3,
            'OPEX': 0
        }
    elif eco_settings.power == 'GG':
        # Ground-gen: Onboard generators
        eco['kite']['obGen'] = {
            'CAPEX': par['kite']['obGen']['p'] * inp['kite']['obGen']['P']/1e3,
            'OPEX': 0
        }

        # Onboard batteries
        eco['kite']['obBatt'] = {
            'CAPEX': par['kite']['obBatt']['p'] * inp['kite']['obBatt']['E'],
            'OPEX': 0
        }

    # Avionics
    eco['kite']['avionics'] = {
        'CAPEX': par['kite']['avio']['C'],
        'OPEX': 0
    }

    return inp, par, eco

"""Ground station subsystem economic calculations.

This module computes the capital and operational expenditures for the ground
station, including winch, drivetrain, generators, storage systems, power
converters, launch/land system, yaw system, and control/communication unit.
"""

import numpy as np
from typing import Dict, Any, Tuple
from .config import eco_settings


def eco_gstation(inp: Dict[str, Any], par: Dict[str, Any], eco: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Calculate costs related to ground station subsystem.

    Args:
        inp: Dictionary containing input parameters.
        par: Dictionary containing cost model parameters.
        eco: Dictionary containing economic results.

    Returns:
        Tuple of (inp, par, eco) dictionaries updated with ground station calculations.
    """
    # Initialize gStation structure in eco if not present
    if 'gStation' not in eco:
        eco['gStation'] = {}

    if eco_settings.power == 'GG':
        # Ground-gen configuration

        # Winch
        eco['gStation']['winch'] = {}
        eco['gStation']['winch']['D'] = (par['gStation']['winch']['dwinch_dt'] *
                                         inp['tether']['d'])

        material = par['gStation']['winch']['material']
        if material == 1:  # Aluminum
            eco['gStation']['winch']['t'] = (np.pi/4 *
                                             par['tether']['sigma_max'] /
                                             par['gStation']['winch']['sigma_al'] *
                                             inp['tether']['d'])
            eco['gStation']['winch']['m'] = (
                np.pi/4 * (eco['gStation']['winch']['D']**2 -
                          (eco['gStation']['winch']['D'] -
                           2*eco['gStation']['winch']['t'])**2) *
                inp['tether']['L'] * inp['tether']['d'] /
                (eco['gStation']['winch']['D']*np.pi) *
                par['gStation']['winch']['rho_al'] *
                par['gStation']['winch']['SF_dt'] *
                par['gStation']['winch']['SF_Lt']
            )
            eco['gStation']['winch']['CAPEX'] = (eco['gStation']['winch']['m'] *
                                                 par['gStation']['winch']['p_al'])
        elif material == 2:  # Steel
            eco['gStation']['winch']['t'] = (np.pi/4 *
                                             par['tether']['sigma_max'] /
                                             par['gStation']['winch']['sigma_st'] *
                                             inp['tether']['d'])
            eco['gStation']['winch']['m'] = (
                np.pi/4 * (eco['gStation']['winch']['D']**2 -
                          (eco['gStation']['winch']['D'] -
                           2*eco['gStation']['winch']['t'])**2) *
                inp['tether']['L'] * inp['tether']['d'] /
                (eco['gStation']['winch']['D']*np.pi) *
                par['gStation']['winch']['rho_st'] *
                par['gStation']['winch']['SF_dt'] *
                par['gStation']['winch']['SF_Lt']
            )
            eco['gStation']['winch']['CAPEX'] = (eco['gStation']['winch']['m'] *
                                                 par['gStation']['winch']['p_st'])

        eco['gStation']['winch']['OPEX'] = 0

        # Drivetrain
        drivetrain_type = par['gStation']['drivetrain_type']

        if drivetrain_type == 1:  # Electric drivetrain
            # Gearbox
            approach = par['gStation']['gearbox']['approach']
            if approach == 1:
                eco['gStation']['gearbox'] = {
                    'CAPEX': (par['gStation']['gearbox']['one']['p'] *
                             inp['system']['P_m_peak']/1e3),
                    'OPEX': 0
                }
            elif approach == 2:
                eco['gStation']['gearbox'] = {}
                eco['gStation']['gearbox']['m'] = (
                    par['gStation']['gearbox']['two']['k'] *
                    (np.max(inp['system']['F_t']) *
                     eco['gStation']['winch']['D']/2/1e3) **
                    par['gStation']['gearbox']['two']['b']
                )
                eco['gStation']['gearbox']['CAPEX'] = (
                    par['gStation']['gearbox']['two']['p'] *
                    eco['gStation']['gearbox']['m']
                )
                eco['gStation']['gearbox']['OPEX'] = 0

            # Electric generator
            approach = par['gStation']['gen']['approach']
            if approach == 1:
                eco['gStation']['gen'] = {
                    'CAPEX': (par['gStation']['gen']['one']['p'] *
                             inp['system']['P_m_peak']/1e3),
                    'OPEX': 0
                }
            elif approach == 2:
                eco['gStation']['gen'] = {}
                eco['gStation']['gen']['m'] = (
                    par['gStation']['gen']['two']['k'] *
                    inp['system']['P_m_peak']/1e3 +
                    par['gStation']['gen']['two']['b']
                )
                eco['gStation']['gen']['CAPEX'] = (
                    par['gStation']['gen']['two']['p'] *
                    eco['gStation']['gen']['m']
                )
                eco['gStation']['gen']['OPEX'] = 0

            # Electrical storage
            elecSto_type = par['gStation']['elecSto_type']
            if elecSto_type == 1:  # Ultracapacitor bank
                eco['gStation']['ultracap'] = {
                    'CAPEX': (par['gStation']['ultracap']['p'] *
                             inp['gStation']['ultracap']['E_rated'])
                }
                if inp['gStation']['ultracap']['f_repl'] < 0:
                    # Calculate replacement frequency
                    E_ex = inp['gStation']['ultracap']['E_ex']
                    Dt_cycle = inp['system']['Dt_cycle']
                    if np.isscalar(E_ex):
                        energy_term = E_ex / Dt_cycle * 3600
                    else:
                        energy_term = E_ex / Dt_cycle * 3600
                        energy_term = np.nan_to_num(energy_term, nan=0.0, posinf=0.0, neginf=0.0)

                    inp['gStation']['ultracap']['f_repl'] = (
                        8760 * np.trapezoid(inp['atm']['gw'] * energy_term,
                                            inp['atm']['wind_range']) /
                        inp['gStation']['ultracap']['E_rated'] /
                        par['gStation']['ultracap']['N']
                    )
                eco['gStation']['ultracap']['OPEX'] = (
                    inp['gStation']['ultracap']['f_repl'] *
                    eco['gStation']['ultracap']['CAPEX']
                )

            elif elecSto_type == 2:  # Battery bank
                eco['gStation']['batt'] = {
                    'CAPEX': (par['gStation']['batt']['p'] *
                             inp['gStation']['batt']['E_rated'])
                }
                if inp['gStation']['batt']['f_repl'] < 0:
                    E_ex = inp['gStation']['batt']['E_ex']
                    Dt_cycle = inp['system']['Dt_cycle']
                    if np.isscalar(E_ex):
                        energy_term = E_ex / Dt_cycle * 3600
                    else:
                        energy_term = E_ex / Dt_cycle * 3600
                        energy_term = np.nan_to_num(energy_term, nan=0.0, posinf=0.0, neginf=0.0)

                    inp['gStation']['batt']['f_repl'] = (
                        8760 * np.trapezoid(inp['atm']['gw'] * energy_term,
                                            inp['atm']['wind_range']) /
                        inp['gStation']['batt']['E_rated'] /
                        par['gStation']['batt']['N']
                    )
                eco['gStation']['batt']['OPEX'] = (
                    inp['gStation']['batt']['f_repl'] *
                    eco['gStation']['batt']['CAPEX']
                )

            # Power converters
            eco['gStation']['powerConv'] = {
                'CAPEX': (par['gStation']['powerConv']['p'] *
                         (inp['system']['P_e_rated'] + inp['system']['P_m_peak'])/1e3),
                'OPEX': 0
            }

        elif drivetrain_type == 2:  # Hydraulic drivetrain
            # Pump Motor
            eco['gStation']['pumpMotor'] = {
                'CAPEX': (par['gStation']['pumpMotor']['p_1'] *
                         inp['system']['P_m_peak']/1e3),
                'OPEX': (inp['gStation']['pumpMotor']['f_repl'] *
                        par['gStation']['pumpMotor']['p_2'] *
                        inp['system']['P_m_peak']/1e3)
            }

            # Hydropneumatic accumulator bank
            eco['gStation']['hydAccum'] = {
                'CAPEX': (par['gStation']['hydAccum']['p_1'] *
                         inp['gStation']['hydAccum']['E_rated']/1e3),
                'OPEX': (inp['gStation']['hydAccum']['f_repl'] *
                        par['gStation']['hydAccum']['p_2'] *
                        inp['gStation']['hydAccum']['E_ex']/1e3)
            }

            # Hydraulic motor
            eco['gStation']['hydMotor'] = {
                'CAPEX': (par['gStation']['hydMotor']['p_1'] *
                         inp['system']['P_e_rated']/1e3),
                'OPEX': (inp['gStation']['hydMotor']['f_repl'] *
                        par['gStation']['hydMotor']['p_2'] *
                        inp['system']['P_e_rated']/1e3)
            }

            # Electric generator
            approach = par['gStation']['gen']['approach']
            if approach == 1:
                eco['gStation']['gen'] = {
                    'CAPEX': (par['gStation']['gen']['one']['p'] *
                             inp['system']['P_e_rated']/1e3),
                    'OPEX': 0
                }
            elif approach == 2:
                eco['gStation']['gen'] = {}
                eco['gStation']['gen']['m'] = (
                    par['gStation']['gen']['two']['k'] *
                    inp['system']['P_e_rated']/1e3 +
                    par['gStation']['gen']['two']['b']
                )
                eco['gStation']['gen']['CAPEX'] = (
                    par['gStation']['gen']['two']['p'] *
                    eco['gStation']['gen']['m']
                )
                eco['gStation']['gen']['OPEX'] = 0

    elif eco_settings.power == 'FG':
        # Fly-gen configuration

        # Winch
        eco['gStation']['winch'] = {}
        eco['gStation']['winch']['D'] = (par['gStation']['winch']['dwinch_dt'] *
                                         inp['tether']['d'])

        material = par['gStation']['winch']['material']
        if material == 1:  # Aluminum
            eco['gStation']['winch']['t'] = inp['tether']['d']
            eco['gStation']['winch']['m'] = (
                np.pi/4 * (eco['gStation']['winch']['D']**2 -
                          (eco['gStation']['winch']['D'] -
                           2*eco['gStation']['winch']['t'])**2) *
                inp['tether']['L'] * inp['tether']['d'] /
                (eco['gStation']['winch']['D']*np.pi) *
                par['gStation']['winch']['rho_al'] *
                par['gStation']['winch']['SF_dt'] *
                par['gStation']['winch']['SF_Lt']
            )
            eco['gStation']['winch']['CAPEX'] = (eco['gStation']['winch']['m'] *
                                                 par['gStation']['winch']['p_al'])
        elif material == 2:  # Steel
            eco['gStation']['winch']['t'] = inp['tether']['d']
            eco['gStation']['winch']['m'] = (
                np.pi/4 * (eco['gStation']['winch']['D']**2 -
                          (eco['gStation']['winch']['D'] -
                           2*eco['gStation']['winch']['t'])**2) *
                inp['tether']['L'] * inp['tether']['d'] /
                (eco['gStation']['winch']['D']*np.pi) *
                par['gStation']['winch']['rho_st'] *
                par['gStation']['winch']['SF_dt'] *
                par['gStation']['winch']['SF_Lt']
            )
            eco['gStation']['winch']['CAPEX'] = (eco['gStation']['winch']['m'] *
                                                 par['gStation']['winch']['p_st'])

        eco['gStation']['winch']['OPEX'] = 0

        # Electrical storage
        elecSto_type = par['gStation']['elecSto_type']
        if elecSto_type == 1:  # Ultracapacitor bank
            eco['gStation']['ultracap'] = {
                'CAPEX': (par['gStation']['ultracap']['p'] *
                         inp['gStation']['ultracap']['E_rated'])
            }
            if inp['gStation']['ultracap']['f_repl'] < 0:
                inp['gStation']['ultracap']['f_repl'] = (
                    8760 * np.trapezoid(inp['atm']['gw'] * inp['atm']['wind_range'] *
                                        inp['system']['lambda'] /
                                        (2 * np.pi * inp['system']['R0']),
                                        inp['atm']['wind_range']) /
                    par['gStation']['ultracap']['N']
                )
            eco['gStation']['ultracap']['OPEX'] = (
                inp['gStation']['ultracap']['f_repl'] *
                eco['gStation']['ultracap']['CAPEX']
            )

        elif elecSto_type == 2:  # Battery bank
            eco['gStation']['batt'] = {
                'CAPEX': (par['gStation']['batt']['p'] *
                         inp['gStation']['batt']['E_rated'])
            }
            if inp['gStation']['batt']['f_repl'] < 0:
                inp['gStation']['batt']['f_repl'] = (
                    8760 * np.trapezoid(inp['atm']['gw'] * inp['atm']['wind_range'] *
                                        inp['system']['lambda'] /
                                        (2 * np.pi * inp['system']['R0']),
                                        inp['atm']['wind_range']) /
                    par['gStation']['batt']['N']
                )
            eco['gStation']['batt']['OPEX'] = (
                inp['gStation']['batt']['f_repl'] *
                eco['gStation']['batt']['CAPEX']
            )

        # Power converters
        eco['gStation']['powerConv'] = {
            'CAPEX': 2 * par['gStation']['powerConv']['p'] * inp['system']['P_e_rated']/1e3,
            'OPEX': 0
        }

    # Common components for both FG and GG

    # Launch & land system
    eco['gStation']['lls'] = {
        'CAPEX': 0,
        'OPEX': 0
    }

    # Yaw system
    eco['gStation']['yaw'] = {
        'CAPEX': 0,
        'OPEX': 0
    }

    # Control and communication unit
    eco['gStation']['controlStation'] = {
        'CAPEX': 0,
        'OPEX': 0
    }

    return inp, par, eco

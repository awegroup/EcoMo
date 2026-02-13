"""Input data generation for ECOMo simulation.

This module provides the eco_system_inputs_example function, which generates
example input parameters for the economic model based on the selected configuration.
"""

import numpy as np
from typing import Dict, Any
from .config import eco_settings
from .eco_import_model import eco_import_model


def eco_system_inputs_example() -> Dict[str, Any]:
    """Generate example input parameters for the economic model.

    Uses the configuration already set in eco_settings (via configure()).
    When input_model_file is 'code', hardcoded values are used.
    Otherwise, values are imported from the corresponding Excel file.

    Returns:
        Dictionary containing input parameters with nested structure.
    """
    inp = {}

    # Common parameters - Weibull wind distribution shape and scale
    k = 2
    A = 8

    # Business related quantities
    inp['business'] = {
        'N_y': 25,       # project years
        'r_d': 0.08,     # cost of debt
        'r_e': 0.12,     # cost of equity
        'TaxRate': 0.25,  # Tax rate (25%)
        'DtoE': 70 / 30   # Debt-Equity-ratio
    }

    # Topology specific parameters
    if eco_settings.input_model_file == 'code':
        # Programmatic input - hardcoded values matching MATLAB 'code' case
        _set_code_inputs(inp, k, A)
    else:
        # Import from Excel file (default MATLAB path)
        inp = eco_import_model(inp)
        # Compute wind distribution from imported wind_range
        wr = inp['atm']['wind_range']
        inp['atm']['gw'] = (k / A * (wr / A) ** (k - 1) *
                            np.exp(-(wr / A) ** k))

    return inp


def _set_code_inputs(inp: Dict[str, Any], k: float, A: float) -> None:
    """Set hardcoded inputs for the 'code' configuration.

    This matches the MATLAB 'case code' branch inside
    eco_system_inputs_example.m.

    Args:
        inp: Dictionary to populate with input parameters.
        k: Weibull shape parameter.
        A: Weibull scale parameter.
    """
    if eco_settings.power == 'FG':
        # Wind resources
        wind_range = np.concatenate([
            np.arange(3, 10 + 1/3, 1/3), [15, 20]
        ])
        gw = (k / A * (wind_range / A) ** (k - 1) *
              np.exp(-(wind_range / A) ** k))

        inp['atm'] = {'wind_range': wind_range, 'gw': gw}

        b = 10  # wingspan

        inp['kite'] = {
            'structure': {
                'm': 122.5, 'b': b, 'AR': 6, 'f_repl': 0
            }
        }

        inp['tether'] = {
            'd': 1.6e-3 * b, 'L': 100, 'rho': 970, 'f_repl': -1
        }

        d = inp['tether']['d']
        F_t = (d ** 2 / 4 * np.pi *
               np.array([0.124572136342289, 0.151797436838344,
                         0.181460994246145, 0.214172304028360,
                         0.250094485666586, 0.289217262316979,
                         0.331480668302760, 0.376825094990467,
                         0.425204519064134, 0.476587107759039,
                         0.530951609064390, 0.588284030406247,
                         0.648575480469101, 0.711819900342773,
                         0.667172308938198, 0.538286443708873,
                         0.479418546822401, 0.440125256353379,
                         0.410771458547833, 0.387467590011167,
                         0.368260306087034, 0.352050875661555,
                         0.3, 0.25]) * 1e9)

        P_e_rated = 100e3
        P_e_avg = P_e_rated * np.array([
            0.0514836036930580, 0.0740924523303653, 0.101398515897930,
            0.133916653105838, 0.172155839837058, 0.216618610982057,
            0.267802451082606, 0.326201238110832, 0.392306298442283,
            0.466607150752925, 0.549592018061346, 0.641748164098206,
            0.743562144341538, 0.855519973731538, 0.978107215387062,
            0.999997934928179, 0.999999974273665, 0.999999480541176,
            0.999999330603523, 0.999999355894903, 0.999999280365251,
            0.999996057827535, 1, 1
        ])

        inp['system'] = {
            'F_t': F_t,
            'P_e_rated': P_e_rated,
            'P_e_avg': P_e_avg,
            'lambda': 7,
            'R0': 5 * b
        }

        E_ultracap = (122.5 * 9.81 * b * 5 / 3.6e6)
        inp['gStation'] = {
            'ultracap': {'E_rated': E_ultracap, 'f_repl': -1},
            'batt': {
                'E_rated': P_e_rated / 1e3, 'f_repl': -1
            }
        }

    elif eco_settings.power == 'GG':
        if eco_settings.wing == 'fixed':
            wind_range = np.arange(1, 26, 1)
            gw = (k / A * (wind_range / A) ** (k - 1) *
                  np.exp(-(wind_range / A) ** k))

            inp['atm'] = {'wind_range': wind_range, 'gw': gw}

            inp['kite'] = {
                'structure': {'m': 5543, 'A': 100, 'f_repl': 0},
                'obGen': {'P': 1e3},
                'obBatt': {'E': 1}
            }
            inp['tether'] = {
                'd': 0.0273, 'L': 2600, 'rho': 970, 'f_repl': -1
            }

            F_t = np.array([
                0, 0, 0, 0, 0, 176218.741942466, 247326.729934132,
                335309.738279319, 349999.999905368, 349999.999983002,
                349999.999999204, 349983.541245218, 348784.001594628,
                345425.348864681, 349095.793762565, 346191.685571336,
                341260.264393564, 340281.862240894, 340333.765419009,
                340876.771496480, 341638.231403760, 342493.800708096,
                343392.451059166, 344311.108894388, 345237.030244762
            ])
            P_e_avg = np.array([
                0, 0, 0, 0, 0, 145524.730974731, 288495.257975699,
                488571.542997977, 727335.279338411, 953941.843495889,
                1000000.00000461, 999999.999798969, 1000000.00001413,
                1000000.00000000, 1000000.00000046, 1000000.00000080,
                1000000.00000000, 1000000.00000017, 1000000.00000000,
                999999.999732790, 999999.999965783, 999999.999995972,
                999999.999999559, 999999.999999956, 1000000.00000000
            ])
            Dt_cycle = np.array([
                0, 0, 0, 0, 0, 130.388949196501, 132.125896939284,
                130.302239127635, 135.689613751230, 132.500820465091,
                124.515660439110, 123.350051661913, 122.919611216260,
                122.626866710798, 122.754436421934, 122.745072840010,
                122.773588271380, 122.901681745870, 123.093919941459,
                123.315575577003, 123.547269528843, 123.778584819469,
                124.003535889111, 124.218500332100, 124.421040233482
            ])

            inp['system'] = {
                'F_t': F_t, 'P_m_peak': 1.87e+06,
                'P_e_avg': P_e_avg, 'P_e_rated': 1e+06,
                'Dt_cycle': Dt_cycle
            }

            E_ex = np.array([
                0, 0, 0, 0, 0, 0.794997961209469, 1.91620710592149,
                3.66340782590143, 7.00053194254757, 10.6616297528835,
                11.1890647627290, 11.2279405187141, 11.2277213125461,
                11.1934366085895, 11.2473922571621, 11.2098142616423,
                11.1438062076693, 11.1273616907102, 11.1228424486107,
                11.1240295153377, 11.1278176039227, 11.1328371206444,
                11.1385755705485, 11.1448282168896, 11.1514961274243
            ])
            inp['gStation'] = {
                'ultracap': {
                    'E_rated': 11.25, 'E_ex': E_ex, 'f_repl': -1
                },
                'batt': {
                    'E_rated': 1e+06 / 1e3, 'E_ex': E_ex, 'f_repl': -1
                },
                'hydAccum': {
                    'E_rated': 11.25, 'E_ex': E_ex, 'f_repl': -1
                },
                'hydMotor': {'f_repl': 0},
                'pumpMotor': {'f_repl': 0}
            }

        elif eco_settings.wing == 'soft':
            wind_range = np.array([
                5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ])
            gw = (k / A * (wind_range / A) ** (k - 1) *
                  np.exp(-(wind_range / A) ** k))

            inp['atm'] = {'wind_range': wind_range, 'gw': gw}

            inp['kite'] = {
                'structure': {'A': 60, 'f_repl': -1},
                'obGen': {'P': 1e3},
                'obBatt': {'E': 0}
            }
            inp['tether'] = {
                'd': 0.008, 'L': 352, 'rho': 970, 'f_repl': -1
            }

            F_t = np.array([
                4083.33333333, 8166.66666667, 12250., 16333.33333333,
                20416.66666667, 24500., 24500., 24500., 24500.,
                24500., 24500., 24500.
            ])
            P_e_avg = 1e3 * np.array([
                1, 5, 10, 20, 28, 30, 30, 30, 30, 30, 29, 28
            ])
            inp['system'] = {
                'F_t': F_t, 'P_m_peak': 40e3,
                'P_e_avg': P_e_avg,
                'P_e_rated': float(np.max(P_e_avg)),
                'Dt_cycle': 100
            }

            E_rated_uc = (40e3 + 10e3) * 20 / 3600 / 1e3
            E_ex = E_rated_uc / 2
            inp['gStation'] = {
                'ultracap': {
                    'E_rated': E_rated_uc, 'E_ex': E_ex,
                    'f_repl': -1
                },
                'batt': {
                    'E_rated': float(np.max(P_e_avg)) / 1e3,
                    'E_ex': E_ex, 'f_repl': -1
                },
                'hydAccum': {
                    'E_rated': E_rated_uc, 'E_ex': E_ex,
                    'f_repl': 0.1
                },
                'hydMotor': {'f_repl': 0.083},
                'pumpMotor': {'f_repl': 0.125}
            }

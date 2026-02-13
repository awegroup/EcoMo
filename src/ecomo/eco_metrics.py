"""Economic metrics calculation.

This module computes various economic metrics such as cost of energy (LCoE),
net present value (NPV), internal rate of return (IRR), return on equity (RoE),
profit, and others.
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Dict, Any, List, Tuple


def _structree(structure: Dict[str, Any],
               path: List[str] = None) -> List[List[str]]:
    """Extract all paths to leaf elements in a nested dictionary.

    Args:
        structure: The nested dictionary to traverse.
        path: Current path (used in recursion).

    Returns:
        List of paths, where each path is a list of keys.
    """
    if path is None:
        path = []
    paths = []
    if isinstance(structure, dict):
        for key, value in structure.items():
            new_path = path + [key]
            if isinstance(value, dict):
                paths.extend(_structree(value, new_path))
            else:
                paths.append(new_path)
    elif path:
        paths.append(path)
    return paths


def eco_npv(r: float, eco: Dict[str, Any], par: Dict[str, Any], N_y: int) -> float:
    """Calculate the net present value (NPV).

    This function is used to compute the internal rate of return (IRR).

    Args:
        r: Discount rate.
        eco: Dictionary containing economic results.
        par: Dictionary containing cost model parameters.
        N_y: Project lifetime in years.

    Returns:
        Net present value.
    """
    NPV = -eco['metrics']['ICC']
    for t in range(1, N_y + 1):
        NPV += ((eco['metrics']['p'] + par['metrics']['subsidy']) *
                eco['metrics']['AEP'] - eco['metrics']['OMC']) / (1 + r)**t
    return NPV


def eco_compute_metrics(inp: Dict[str, Any], par: Dict[str, Any], eco: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Calculate economic metrics for ECOMo simulation.

    This function computes various economic metrics such as LCoE, NPV,
    RoE, profit, and others.

    Args:
        inp: Dictionary containing input parameters.
        par: Dictionary containing cost model parameters.
        eco: Dictionary containing economic results.

    Returns:
        Tuple of (inp, par, eco) dictionaries updated with economic metrics.
    """
    # Initialize metrics structure
    if 'metrics' not in eco:
        eco['metrics'] = {}

    # WACC (Weighted Average Cost of Capital)
    inp['business']['r'] = (
        inp['business']['DtoE'] / (1 + inp['business']['DtoE']) *
        inp['business']['r_d'] * (1 - inp['business']['TaxRate']) +
        1 / (1 + inp['business']['DtoE']) * inp['business']['r_e']
    )

    r = inp['business']['r']
    N_y = inp['business']['N_y']

    # Capital Recovery Factor
    eco['metrics']['CRF'] = r * (1 + r)**N_y / ((1 + r)**N_y - 1)

    # Annual Energy Production (MWh)
    eco['metrics']['AEP'] = (
        8760 * np.trapezoid(inp['system']['P_e_avg'] * inp['atm']['gw'],
                            inp['atm']['wind_range']) / 1e6
    )

    # Capacity Factor
    eco['metrics']['CF'] = (
        eco['metrics']['AEP'] / (inp['system']['P_e_rated']/1e6 * 8760)
    )

    # Average electricity price (euros/MWh)
    eco['metrics']['p'] = (
        8760 * np.trapezoid(
            (par['metrics']['electricity']['p_0'] +
             par['metrics']['electricity']['p_1'] * inp['atm']['wind_range']) *
            inp['system']['P_e_avg'] / 1e6 * inp['atm']['gw'],
            inp['atm']['wind_range']) /
        eco['metrics']['AEP']
    )

    # Average electricity price without weighting (euros/MWh)
    eco['metrics']['p_hat'] = np.trapezoid(
        (par['metrics']['electricity']['p_0'] +
         par['metrics']['electricity']['p_1'] * inp['atm']['wind_range']) *
        inp['atm']['gw'],
        inp['atm']['wind_range']
    )

    # Value factor
    eco['metrics']['vf'] = eco['metrics']['p'] / eco['metrics']['p_hat']

    # Find all CAPEX and OPEX and organize them
    PATH = _structree(eco)
    eco['metrics']['ICC'] = 0  # Initial Capital Cost (sum of total CAPEX)
    eco['metrics']['OMC'] = 0  # Operational Maintenance Cost (sum of total OPEX)

    eco['metrics']['icc'] = []
    eco['metrics']['icc_name'] = []
    eco['metrics']['omc'] = []
    eco['metrics']['omc_name'] = []
    eco['metrics']['LCoE_contr'] = []
    eco['metrics']['LCoE_contr_name'] = []

    for path in PATH:
        if path[-1] == 'CAPEX':
            # Extract value
            current = eco
            for key in path:
                current = current[key]
            value = current

            # Create name
            if len(path) == 2:
                name = path[0]
            else:
                name = '.'.join(path[:-1])

            eco['metrics']['icc'].append(value)
            eco['metrics']['icc_name'].append(name)
            eco['metrics']['ICC'] += value
            eco['metrics']['LCoE_contr'].append(
                value * eco['metrics']['CRF'] / eco['metrics']['AEP']
            )
            eco['metrics']['LCoE_contr_name'].append(name + '.capex')

        elif path[-1] == 'OPEX':
            # Extract value
            current = eco
            for key in path:
                current = current[key]
            value = current

            # Create name
            if len(path) == 2:
                name = path[0]
            else:
                name = '.'.join(path[:-1])

            eco['metrics']['omc'].append(value)
            eco['metrics']['omc_name'].append(name)
            eco['metrics']['OMC'] += value
            eco['metrics']['LCoE_contr'].append(value / eco['metrics']['AEP'])
            eco['metrics']['LCoE_contr_name'].append(name + '.opex')

    # Cash flow
    eco['metrics']['cashflow'] = np.zeros(N_y + 1)
    eco['metrics']['cashflow'][0] = -eco['metrics']['ICC']  # Year before start
    for t in range(1, N_y + 1):
        eco['metrics']['cashflow'][t] = (
            (eco['metrics']['p'] + par['metrics']['subsidy']) *
            eco['metrics']['AEP'] - eco['metrics']['OMC']
        )

    # Calculate cumulative cashflow
    cumulative_cashflow = np.cumsum(eco['metrics']['cashflow'])

    # Find the payback year
    payback_indices = np.where(cumulative_cashflow >= 0)[0]
    if len(payback_indices) > 0:
        eco['metrics']['payback_year'] = payback_indices[0] - 1
    else:
        eco['metrics']['payback_year'] = N_y  # Project doesn't pay back

    # Compute metrics
    num_LCoE = eco['metrics']['ICC']
    num_LRoE = 0
    den = 0
    den_cove = 0

    for t in range(1, N_y + 1):
        num_LCoE += eco['metrics']['OMC'] / (1 + r)**t
        num_LRoE += ((eco['metrics']['p'] + par['metrics']['subsidy']) *
                     eco['metrics']['AEP'] / (1 + r)**t)
        den += eco['metrics']['AEP'] / (1 + r)**t
        den_cove += eco['metrics']['vf'] * eco['metrics']['AEP'] / (1 + r)**t

    # NPV
    eco['metrics']['NPV'] = eco_npv(r, eco, par, N_y)

    # IRR (Internal Rate of Return)
    try:
        result = fsolve(lambda r_irr: eco_npv(r_irr, eco, par, N_y), 0)
        eco['metrics']['IRR'] = result[0]
    except:
        eco['metrics']['IRR'] = np.nan

    # LCoE (Levelized Cost of Energy)
    eco['metrics']['LCoE'] = num_LCoE / den

    # CoVE (Cost of Valued Energy)
    eco['metrics']['CoVE'] = num_LCoE / den_cove

    # LRoE (Levelized Revenue of Energy)
    eco['metrics']['LRoE'] = num_LRoE / den

    # LPoE (Levelized Profit of Energy)
    eco['metrics']['LPoE'] = eco['metrics']['LRoE'] - eco['metrics']['LCoE']

    # Pi (Annual Profit)
    eco['metrics']['Pi'] = eco['metrics']['LPoE'] * eco['metrics']['AEP']

    return inp, par, eco

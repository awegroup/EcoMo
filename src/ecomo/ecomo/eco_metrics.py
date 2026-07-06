"""Economic metrics calculation.

This module computes various economic metrics such as cost of energy
(LCoE), net present value (NPV), internal rate of return (IRR), return
on equity (RoE), profit, and others.

The annual energy production (AEP) is normally integrated from the
power curve and the wind distribution. When an externally computed AEP
is available (e.g. from an AWESPA ``aep_results.yml`` file), that value
is used directly instead.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Dict, Any, List

from .constants import HOURS_PER_YEAR, W_PER_MW
from .eco_costs import MarketCosts
from .eco_inputs import BusinessInputs, PerformanceData


def _structree(structure: Dict[str, Any],
               path: List[str] = None) -> List[List[str]]:
    """Extract all paths to leaf elements in a nested dictionary.

    Args:
        structure (dict): The nested dictionary to traverse.
        path (list): Current path (used in recursion).

    Returns:
        list: List of paths, where each path is a list of keys.
    """
    if path is None:
        path = []
    paths = []
    if isinstance(structure, dict):
        for key, value in structure.items():
            newPath = path + [key]
            if isinstance(value, dict):
                paths.extend(_structree(value, newPath))
            else:
                paths.append(newPath)
    elif path:
        paths.append(path)
    return paths


def eco_npv(r: float, metrics: Dict[str, Any], subsidy: float,
            nY: int) -> float:
    """Calculate the net present value (NPV).

    This function is also used to compute the internal rate of return
    (IRR).

    Args:
        r (float): Discount rate.
        metrics (dict): Metrics with ``ICC``, ``p``, ``AEP`` and
            ``OMC``.
        subsidy (float): Production subsidy [EUR/MWh].
        nY (int): Project lifetime in years.

    Returns:
        float: Net present value.
    """
    netRevenue = ((metrics['p'] + subsidy) * metrics['AEP'] - metrics['OMC'])
    years = np.arange(1, nY + 1)
    return -metrics['ICC'] + np.sum(netRevenue / (1 + r) ** years)


def eco_irr(metrics: Dict[str, Any], subsidy: float, nY: int) -> float:
    """Calculate the internal rate of return (IRR).

    The IRR is the discount rate at which the project NPV is zero.
    With a single upfront investment and a constant annual net
    revenue, the NPV decreases monotonically with the discount rate,
    so the root is unique and found by bracketed root finding.

    Args:
        metrics (dict): Metrics with ``ICC``, ``p``, ``AEP`` and
            ``OMC``.
        subsidy (float): Production subsidy [EUR/MWh].
        nY (int): Project lifetime in years.

    Returns:
        float: Internal rate of return, or NaN when the annual net
        revenue is not positive (the NPV never crosses zero).
    """
    netRevenue = ((metrics['p'] + subsidy) * metrics['AEP'] - metrics['OMC'])
    if not np.isfinite(netRevenue) or netRevenue <= 0:
        return np.nan

    # NPV -> +inf as r -> -1 and NPV -> -ICC as r -> +inf; expand the
    # upper bound until the root is bracketed
    lowerRate = -0.99
    upperRate = 1.0
    while eco_npv(upperRate, metrics, subsidy, nY) > 0:
        upperRate *= 2
        if upperRate > 1e9:
            return np.nan
    if eco_npv(lowerRate, metrics, subsidy, nY) < 0:
        return np.nan

    return brentq(lambda r: eco_npv(r, metrics, subsidy, nY),
                  lowerRate, upperRate)


def _electricity_price(windSpeeds: np.ndarray,
                       market: MarketCosts) -> np.ndarray:
    """Electricity price as a function of wind speed [EUR/MWh].

    Args:
        windSpeeds (np.ndarray): Wind speeds [m/s].
        market (MarketCosts): Market parameters.

    Returns:
        np.ndarray: Electricity price at each wind speed.
    """
    return (market.electricityPriceIntercept +
            market.electricityPriceSlope * windSpeeds)


def _accumulate_costs(eco: Dict[str, Any], metrics: Dict[str, Any],
                      crf: float, aep: float) -> None:
    """Sum the CAPEX/OPEX leaves of the subsystem results.

    Populates ``metrics`` with the total initial capital cost (ICC),
    total operational maintenance cost (OMC) and the per-component
    contribution lists used by the breakdown and the plots.

    Args:
        eco (dict): Assembled subsystem results (without metrics).
        metrics (dict): Metrics dictionary to update in place.
        crf (float): Capital recovery factor [-].
        aep (float): Annual energy production [MWh].
    """
    metrics['ICC'] = 0
    metrics['OMC'] = 0
    metrics['icc'] = []
    metrics['icc_name'] = []
    metrics['omc'] = []
    metrics['omc_name'] = []
    metrics['LCoE_contr'] = []
    metrics['LCoE_contr_name'] = []

    for path in _structree(eco):
        if path[-1] not in ('CAPEX', 'OPEX'):
            continue
        value = eco
        for key in path:
            value = value[key]
        name = path[0] if len(path) == 2 else '.'.join(path[:-1])

        if path[-1] == 'CAPEX':
            metrics['icc'].append(value)
            metrics['icc_name'].append(name)
            metrics['ICC'] += value
            metrics['LCoE_contr'].append(value * crf / aep)
            metrics['LCoE_contr_name'].append(name + '.capex')
        else:
            metrics['omc'].append(value)
            metrics['omc_name'].append(name)
            metrics['OMC'] += value
            metrics['LCoE_contr'].append(value / aep)
            metrics['LCoE_contr_name'].append(name + '.opex')


def eco_compute_metrics(
    eco: Dict[str, Any],
    business: BusinessInputs,
    performance: PerformanceData,
    market: MarketCosts,
    availability: float = 1.0,
) -> Dict[str, Any]:
    """Calculate the economic metrics for the ECOMo simulation.

    Computes various economic metrics such as LCoE, NPV, RoE, profit,
    and others. If ``performance.externalAep`` is set (AEP in MWh), it
    is used directly instead of integrating the power curve over the
    wind distribution. The production-weighted electricity price ``p``
    is always normalized with the integrated production, so it stays a
    proper weighted average when an external AEP is provided.

    The ``availability`` (fraction of the operating-wind time the system
    is actually flown) scales the delivered energy, so the net AEP is
    ``a * AEP_gross``. This keeps downtime consistent: it reduces both
    the revenue/energy and the operating hours (hence the labour and
    load-driven O&M scale with the same ``a``). Baseline ``a = 1`` (no
    downtime) leaves the result unchanged.

    Args:
        eco (dict): Assembled subsystem results (without metrics).
        business (BusinessInputs): Financial parameters.
        performance (PerformanceData): System performance data.
        market (MarketCosts): Market and electricity price parameters.
        availability (float): Fraction of the operating-wind time flown
            [-]; scales the net AEP. Defaults to 1.0.

    Returns:
        dict: The ``eco['metrics']`` results subtree.
    """
    metrics: Dict[str, Any] = {}

    r = business.wacc
    nY = business.nYears
    windSpeeds = performance.windSpeeds
    windPdf = performance.windPdf
    electricityPrice = _electricity_price(windSpeeds, market)

    # Capital Recovery Factor
    metrics['CRF'] = r * (1 + r) ** nY / ((1 + r) ** nY - 1)

    # Annual Energy Production (MWh); the power curve integral is also
    # the weight normalization of the electricity price below
    integratedAep = (HOURS_PER_YEAR *
                     np.trapezoid(performance.averagePower * windPdf,
                                  windSpeeds) / W_PER_MW)
    grossAep = (float(performance.externalAep)
                if performance.externalAep is not None
                else integratedAep)
    # Net AEP after downtime; the price weighting below keeps the gross
    # integral as its (intensive) normalization, so ``p`` is unchanged.
    metrics['AEP'] = availability * grossAep

    # Capacity Factor (on the net, delivered energy)
    metrics['CF'] = (metrics['AEP'] /
                     (performance.ratedPower / W_PER_MW * HOURS_PER_YEAR))

    # Production-weighted average electricity price (EUR/MWh)
    metrics['p'] = (HOURS_PER_YEAR *
                    np.trapezoid(electricityPrice * performance.averagePower /
                                 W_PER_MW * windPdf, windSpeeds) /
                    integratedAep)

    # Unweighted average electricity price (EUR/MWh)
    metrics['p_hat'] = np.trapezoid(electricityPrice * windPdf, windSpeeds)

    # Value factor
    metrics['vf'] = metrics['p'] / metrics['p_hat']

    # Sum the CAPEX/OPEX of the subsystem results
    _accumulate_costs(eco, metrics, metrics['CRF'], metrics['AEP'])

    # Cash flow: the investment in year zero followed by a constant
    # annual net revenue
    annualNetRevenue = ((metrics['p'] + market.subsidy) * metrics['AEP'] -
                        metrics['OMC'])
    metrics['cashflow'] = np.full(nY + 1, annualNetRevenue)
    metrics['cashflow'][0] = -metrics['ICC']  # Year before the start

    # Payback year
    cumulativeCashflow = np.cumsum(metrics['cashflow'])
    paybackIndices = np.where(cumulativeCashflow >= 0)[0]
    metrics['payback_year'] = (paybackIndices[0] - 1
                               if len(paybackIndices) > 0 else nY)

    # All yearly terms are constant, so the sums reduce to one
    # discount-factor sum over the project lifetime
    discountSum = np.sum((1 + r) ** -np.arange(1, nY + 1))
    numLcoe = metrics['ICC'] + metrics['OMC'] * discountSum
    numLroe = (metrics['p'] + market.subsidy) * metrics['AEP'] * discountSum
    den = metrics['AEP'] * discountSum
    denCove = metrics['vf'] * metrics['AEP'] * discountSum

    metrics['NPV'] = eco_npv(r, metrics, market.subsidy, nY)
    metrics['IRR'] = eco_irr(metrics, market.subsidy, nY)
    metrics['LCoE'] = numLcoe / den
    metrics['CoVE'] = numLcoe / denCove
    metrics['LRoE'] = numLroe / den
    metrics['LPoE'] = metrics['LRoE'] - metrics['LCoE']
    metrics['Pi'] = metrics['LPoE'] * metrics['AEP']

    return metrics

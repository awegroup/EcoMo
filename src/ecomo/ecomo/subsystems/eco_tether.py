"""Tether subsystem economic calculations.

This module computes the cost and operational parameters related to the
tether, including cross-sectional area, CAPEX, OPEX, and life
estimation.
"""

import numpy as np
from typing import Dict, Any

from ..constants import (
    BEND_LIFE_CORRECTION,
    HOURS_PER_YEAR,
    PA_PER_GPA,
    SECONDS_PER_HOUR,
)
from ..eco_costs import TetherCosts
from ..eco_inputs import BusinessInputs, PerformanceData, TetherInputs, Topology


def _bending_replacement_frequency(
    tetherStress: np.ndarray,
    performance: PerformanceData,
    costs: TetherCosts,
) -> float:
    """Estimate the GG tether replacement frequency due to bending.

    Args:
        tetherStress (np.ndarray): Fibre stress per wind speed [Pa].
        performance (PerformanceData): System performance data.
        costs (TetherCosts): Tether cost parameters.

    Returns:
        float: Replacement frequency due to bending fatigue [1/year].
    """
    bendExponent = (costs.bendingLifeA1 -
                    costs.bendingLifeA2 * tetherStress / PA_PER_GPA)
    nBends = 10 ** bendExponent
    with np.errstate(divide='ignore', invalid='ignore'):
        integralTerm = (performance.windPdf /
                        (performance.cycleTime / HOURS_PER_YEAR /
                         SECONDS_PER_HOUR * nBends))
    integralTerm = np.nan_to_num(integralTerm, nan=0.0,
                                 posinf=0.0, neginf=0.0)
    lifeBend = 1 / (costs.nBends *
                    np.trapezoid(integralTerm, performance.windSpeeds))
    return 1 / lifeBend / BEND_LIFE_CORRECTION


def _creep_replacement_frequency(
    tetherStress: np.ndarray,
    performance: PerformanceData,
    costs: TetherCosts,
) -> float:
    """Estimate the tether replacement frequency due to creep.

    Args:
        tetherStress (np.ndarray): Fibre stress per wind speed [Pa].
        performance (PerformanceData): System performance data.
        costs (TetherCosts): Tether cost parameters.

    Returns:
        float: Replacement frequency due to creep rupture [1/year].
    """
    creepCoef = costs.creepLifeCoefficients
    if np.isscalar(creepCoef) and not isinstance(creepCoef, np.ndarray):
        lifeCreepArray = creepCoef * np.ones_like(tetherStress)
    else:
        creepExponent = np.polyval(creepCoef, tetherStress / PA_PER_GPA)
        lifeCreepArray = 10 ** creepExponent
    lifeCreep = 1 / np.trapezoid(performance.windPdf / lifeCreepArray,
                                 performance.windSpeeds)
    return 1 / lifeCreep


def eco_tether(
    tether: TetherInputs,
    performance: PerformanceData,
    costs: TetherCosts,
    business: BusinessInputs,
    topology: Topology,
) -> Dict[str, Any]:
    """Calculate cost and operational parameters related to the tether.

    Args:
        tether (TetherInputs): Tether parameters.
        performance (PerformanceData): System performance data.
        costs (TetherCosts): Tether cost parameters.
        business (BusinessInputs): Financial parameters (for the
            project lifetime).
        topology (Topology): System topology.

    Returns:
        dict: The ``eco['tether']`` results subtree.
    """
    eco: Dict[str, Any] = {}

    # CAPEX; the FG tether carries an additional manufacturing factor
    baseCapex = (costs.priceMass * tether.area * costs.fibreAreaFraction *
                 tether.length * tether.density *
                 (1 + costs.coatingMassFraction))
    if topology.power == 'GG':
        eco['CAPEX'] = baseCapex
    elif topology.power == 'FG':
        eco['CAPEX'] = costs.conductiveManufacturingFactor * baseCapex

    # Tether fibre stress
    tetherStress = np.minimum(
        performance.tetherForce / (costs.fibreAreaFraction * tether.area),
        costs.maxStress)
    eco['sigma'] = tetherStress

    # OPEX: replacement frequency from the bending (GG) and creep
    # (GG and FG) life models
    replacementBend = None
    if topology.power == 'GG':
        replacementBend = _bending_replacement_frequency(
            tetherStress, performance, costs)
        eco['f_repl_bend'] = replacementBend
    replacementCreep = _creep_replacement_frequency(
        tetherStress, performance, costs)
    eco['f_repl_creep'] = replacementCreep

    if tether.replacementFrequency is None:
        if topology.power == 'GG':
            replacementFrequency = max(replacementBend, replacementCreep)
        else:  # FG
            replacementFrequency = replacementCreep
    else:
        replacementFrequency = tether.replacementFrequency

    # A tether life beyond the project lifetime means no replacement
    # (a replacement frequency of zero already means an infinite life)
    if (replacementFrequency == 0 or
            1 / replacementFrequency > business.nYears):
        replacementFrequency = 0

    eco['f_repl'] = replacementFrequency
    eco['OPEX'] = replacementFrequency * eco['CAPEX']

    return eco

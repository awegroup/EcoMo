"""Tether subsystem economic calculations.

This module computes the cost and operational parameters related to the
tether, including cross-sectional area, CAPEX, OPEX, and life
estimation.
"""

import numpy as np
from typing import Dict, Any

from ..constants import (
    HOURS_PER_YEAR,
    PA_PER_GPA,
    PA_PER_MPA,
    SECONDS_PER_HOUR,
)
from ..eco_costs import TetherCosts
from ..eco_hours import annual_flight_hours
from ..eco_inputs import BusinessInputs, PerformanceData, TetherInputs, Topology


def _cycles_to_failure(tetherStress: np.ndarray,
                       costs: TetherCosts) -> np.ndarray:
    """Bending cycles to failure per wind speed [-].

    Uses the HMPE bearing-pressure master curve when configured
    (``masterCurveCoeff`` set), otherwise the legacy a1/a2 semi-log S-N
    model. See reports/COST_MODEL_REFERENCE.md for the equations.

    Args:
        tetherStress (np.ndarray): Fibre stress per wind speed [Pa].
        costs (TetherCosts): Tether cost parameters.

    Returns:
        np.ndarray: Cycles to failure per wind speed.
    """
    if costs.masterCurveCoeff is not None:
        sigmaMpa = tetherStress / PA_PER_MPA
        bearingPressure = (costs.bearingPressureCoeff * sigmaMpa /
                           costs.bendingDdRatio)
        if costs.pwLimitMpa is not None:
            bearingPressure = np.maximum(bearingPressure, costs.pwLimitMpa)
        return costs.masterCurveCoeff * bearingPressure ** (
            -costs.masterCurveExponent)
    bendExponent = (costs.bendingLifeA1 -
                    costs.bendingLifeA2 * tetherStress / PA_PER_GPA)
    return 10 ** bendExponent


def _bending_replacement_frequency(
    tetherStress: np.ndarray,
    performance: PerformanceData,
    costs: TetherCosts,
) -> float:
    """Estimate the GG tether replacement frequency due to bending.

    A design retirement safety factor, when configured, scales the
    resulting frequency up (retire before failure).

    Args:
        tetherStress (np.ndarray): Fibre stress per wind speed [Pa].
        performance (PerformanceData): System performance data.
        costs (TetherCosts): Tether cost parameters.

    Returns:
        float: Replacement frequency due to bending fatigue [1/year].
    """
    nFail = _cycles_to_failure(tetherStress, costs)
    with np.errstate(divide='ignore', invalid='ignore'):
        integralTerm = (performance.windPdf /
                        (performance.cycleTime / HOURS_PER_YEAR /
                         SECONDS_PER_HOUR * nFail))
    integralTerm = np.nan_to_num(integralTerm, nan=0.0,
                                 posinf=0.0, neginf=0.0)
    lifeBend = 1 / (costs.nBends *
                    np.trapezoid(integralTerm, performance.windSpeeds))
    frequency = 1 / lifeBend
    # The safety factor derates only the master-curve life, not the a1/a2
    # fallback (see reports/COST_MODEL_REFERENCE.md).
    if (costs.masterCurveCoeff is not None and
            costs.designSafetyFactor is not None):
        frequency = costs.designSafetyFactor * frequency
    return frequency


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


def _operational_replacement_frequency(
    performance: PerformanceData,
    costs: TetherCosts,
    availability: float,
) -> float:
    """Estimate the tether replacement frequency due to operational wear.

    Non-fatigue degradation (UV, abrasion, particle ingress, handling) is
    not stress-driven, so it is modelled as an empirical life in flight
    hours consumed at the annual flight-hour rate.

    Args:
        performance (PerformanceData): System performance data.
        costs (TetherCosts): Tether cost parameters.
        availability (float): Fraction of operating-wind time flown [-].

    Returns:
        float: Replacement frequency due to operational wear [1/year].
    """
    return annual_flight_hours(performance, availability) / costs.operationalLife


def eco_tether(
    tether: TetherInputs,
    performance: PerformanceData,
    costs: TetherCosts,
    business: BusinessInputs,
    topology: Topology,
    availability: float = 1.0,
) -> Dict[str, Any]:
    """Calculate cost and operational parameters related to the tether.

    Args:
        tether (TetherInputs): Tether parameters.
        performance (PerformanceData): System performance data.
        costs (TetherCosts): Tether cost parameters.
        business (BusinessInputs): Financial parameters (for the
            project lifetime).
        topology (Topology): System topology.
        availability (float): Fraction of the operating-wind time the
            system is flown [-], used by the operational-wear life model.
            Defaults to 1.0.

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

    # OPEX: the governing mode is the shortest life (highest replacement
    # frequency) of bending, creep and the optional operational wear.
    modeFrequencies: Dict[str, float] = {}
    replacementBend = None
    if topology.power == 'GG':
        replacementBend = _bending_replacement_frequency(
            tetherStress, performance, costs)
        eco['f_repl_bend'] = replacementBend
        modeFrequencies['bending'] = replacementBend
    replacementCreep = _creep_replacement_frequency(
        tetherStress, performance, costs)
    eco['f_repl_creep'] = replacementCreep
    modeFrequencies['creep'] = replacementCreep
    # Operational wear is opt-in (operationalLife is None when disabled).
    if costs.operationalLife is not None:
        replacementOper = _operational_replacement_frequency(
            performance, costs, availability)
        eco['f_repl_oper'] = replacementOper
        modeFrequencies['operational'] = replacementOper

    if tether.replacementFrequency is None:
        governingMode = max(modeFrequencies, key=modeFrequencies.get)
        governingFrequency = modeFrequencies[governingMode]
    else:
        governingMode = 'override'
        governingFrequency = tether.replacementFrequency

    # Physical life from the governing mode, reported before the project-life
    # cap below so a life longer than the project stays visible.
    flightHours = annual_flight_hours(performance, availability)
    eco['life'] = {
        'governing_mode': governingMode,
        'life_flight_hours': (flightHours / governingFrequency
                              if governingFrequency > 0 else None),
        'life_years': (1.0 / governingFrequency
                       if governingFrequency > 0 else None),
        'annual_flight_hours': flightHours,
    }
    # Per-mode life in flight hours (diagnostic)
    for name, freq in modeFrequencies.items():
        eco['life'][f'{name}_flight_hours'] = (
            flightHours / freq if freq > 0 else None)

    # A life beyond the project lifetime charges no replacement.
    replacementFrequency = governingFrequency
    if (replacementFrequency == 0 or
            1 / replacementFrequency > business.nYears):
        replacementFrequency = 0

    eco['f_repl'] = replacementFrequency
    eco['OPEX'] = replacementFrequency * eco['CAPEX']

    return eco

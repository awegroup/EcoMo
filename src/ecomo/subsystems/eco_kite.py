"""Kite subsystem economic calculations.

This module computes the capital expenditure (CAPEX) and operational
expenditure (OPEX) associated with the kite subsystem, including
structure, onboard generators, onboard batteries, and avionics.
"""

import warnings
import numpy as np
from typing import Dict, Any

from ..constants import KiteStructureCostModel, W_PER_KW
from ..eco_costs import KiteCosts
from ..eco_hours import (
    annual_cycle_count,
    annual_flight_hours,
    annual_reelout_hours,
)
from ..eco_inputs import BusinessInputs, KiteInputs, PerformanceData, Topology


def _soft_structure_capex(kite: KiteInputs, costs: KiteCosts) -> float:
    """Compute the soft-wing structure CAPEX [EUR].

    Uses the two-term scaling model when its parameters are set,
    otherwise the original flat-price model.

    Args:
        kite (KiteInputs): Kite parameters.
        costs (KiteCosts): Kite cost parameters.

    Returns:
        float: Structure CAPEX [EUR].
    """
    if costs.materialCostRef is not None:
        return (costs.materialCostRef *
                (kite.flatArea / costs.referenceArea) **
                costs.materialScalingExponent +
                costs.labourCostCoefficient * kite.flatArea)
    return (costs.priceFabric + costs.priceBridle) * kite.flatArea


def _avionics_capex(kite: KiteInputs, costs: KiteCosts) -> float:
    """Compute the avionics (KCU) CAPEX [EUR].

    Uses the area-scaled KCU model when its parameters are set,
    otherwise the flat avionics cost.

    Args:
        kite (KiteInputs): Kite parameters.
        costs (KiteCosts): Kite cost parameters.

    Returns:
        float: Avionics CAPEX [EUR].
    """
    if costs.avionicsCostFixed is not None:
        return (costs.avionicsCostFixed +
                costs.avionicsCostVarRef *
                (kite.flatArea / costs.avionicsReferenceArea) **
                costs.avionicsScalingExponent)
    return costs.avionicsCost


def _soft_structure_replacement_frequency(
    kite: KiteInputs,
    costs: KiteCosts,
    performance: PerformanceData,
    availability: float,
) -> float:
    """Estimate the soft-wing structure replacement frequency.

    Dispatches between two models:

    - Reel-out-hour model (preferred, when ``canopyLifetimeFlightHours``
      is set): the canopy is consumed by accumulated reel-out (traction)
      hours. This scales with how much the kite actually operates and
      matches the physical intuition that the canopy is loaded only on
      reel-out.
    - Legacy calendar model (fallback): a stress-weighted loading factor
      divided by a calendar structural lifetime.

    Args:
        kite (KiteInputs): Kite parameters.
        costs (KiteCosts): Kite cost parameters.
        performance (PerformanceData): System performance data.
        availability (float): Fraction of operating-wind time flown [-].

    Returns:
        float: Replacement frequency [1/year].
    """
    if costs.canopyLifetimeFlightHours is not None:
        return _reelout_hour_replacement_frequency(
            costs, performance, availability)
    return _calendar_replacement_frequency(kite, costs, performance)


def _canopy_load_weight(costs: KiteCosts,
                        performance: PerformanceData):
    """Per-wind-speed S-N load weight ``(F/F_ref)^m`` for the canopy.

    Applies a Miner's-rule damage weighting so that partial-load reel-out
    hours consume less canopy life than peak-load hours. Returns None when
    no load weighting is requested (``canopyLoadExponent`` unset or zero)
    or when no tether force is available (with a warning), in which case
    the plain, load-independent hour count is used.

    Args:
        costs (KiteCosts): Kite cost parameters.
        performance (PerformanceData): System performance data.

    Returns:
        np.ndarray: The per-wind-speed weight, or None for no weighting.
    """
    exponent = costs.canopyLoadExponent
    if not exponent:
        return None

    # Structural fatigue accrues during the traction (reel-out) phase, so
    # prefer the traction-phase force when available.
    force = (performance.tractionTetherForce
             if performance.tractionTetherForce is not None
             else performance.tetherForce)
    force = None if force is None else np.asarray(force, dtype=float)
    if force is None or not np.any(force > 0):
        warnings.warn(
            "canopy_load_exponent is set but no tether force is available; "
            "using unweighted reel-out hours (load-independent canopy "
            "life).",
            UserWarning, stacklevel=3,
        )
        return None

    referenceForce = costs.canopyReferenceForce or float(np.max(force))
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = (force / referenceForce) ** exponent
    return np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)


def _reelout_hour_replacement_frequency(
    costs: KiteCosts,
    performance: PerformanceData,
    availability: float,
) -> float:
    """Reel-out-hour soft-wing structure replacement frequency.

    The canopy is consumed by accumulated reel-out (traction) hours, so
    the replacement frequency is the annual reel-out hours divided by
    ``canopyLifetimeFlightHours``, plus an optional per-cycle wear term.
    An optional load exponent weights the hours by ``(F/F_ref)^m``
    (Miner's rule). See reports/COST_MODEL_REFERENCE.md.

    Args:
        costs (KiteCosts): Kite cost parameters.
        performance (PerformanceData): System performance data.
        availability (float): Fraction of operating-wind time flown [-].

    Returns:
        float: Replacement frequency [1/year].
    """
    loadWeight = _canopy_load_weight(costs, performance)
    reelOutHours = annual_reelout_hours(performance, availability,
                                        load_weight=loadWeight)
    if reelOutHours is None:
        warnings.warn(
            "reel-out time fraction not available; counting all flight "
            "hours as loaded (reel-out fraction = 1) for the canopy "
            "replacement frequency. Provide 'reel_out_time' in the power "
            "curves for the reel-out-weighted estimate.",
            UserWarning, stacklevel=3,
        )
        reelOutHours = annual_flight_hours(performance, availability)

    frequency = reelOutHours / costs.canopyLifetimeFlightHours
    if costs.perCyclePenalty:
        frequency += (costs.perCyclePenalty *
                      annual_cycle_count(performance, availability))
    return frequency


def _calendar_replacement_frequency(
    kite: KiteInputs,
    costs: KiteCosts,
    performance: PerformanceData,
) -> float:
    """Legacy calendar soft-wing structure replacement frequency.

    The equivalent used lifetime per year follows from the loading
    factor (the wind-distribution-weighted tether force relative to the
    maximum force) divided by the structural lifetime at full loading.
    For a pumping kite the structure is only loaded during the reel-out
    phase, so the loading factor is weighted by the reel-out time
    fraction when that timing data is available.

    Args:
        kite (KiteInputs): Kite parameters.
        costs (KiteCosts): Kite cost parameters.
        performance (PerformanceData): System performance data.

    Returns:
        float: Replacement frequency [1/year].
    """
    # Structural fatigue accrues during the traction (reel-out) phase,
    # so prefer the traction-phase force when available.
    force = (performance.tractionTetherForce
             if performance.tractionTetherForce is not None
             else performance.tetherForce)
    relativeForce = force / np.max(force)
    if (performance.reelOutTimeFraction is not None and
            np.any(performance.reelOutTimeFraction > 0)):
        # Time-weighted loading factor: only the reel-out phase loads the
        # kite (the unweighted form overestimates it for pumping kites).
        loadFactor = np.trapezoid(
            performance.windPdf * relativeForce *
            performance.reelOutTimeFraction,
            performance.windSpeeds)
    else:
        warnings.warn(
            "reel-out time fraction not available; using the unweighted "
            "loading factor (Joshi & Trevisi 2024, Eq. 4). This "
            "overestimates the soft-kite replacement frequency by ~38% "
            "for pumping kites. Provide 'reel_out_time' in the power "
            "curves to use the corrected time-weighted formula.",
            UserWarning, stacklevel=2,
        )
        loadFactor = np.trapezoid(
            performance.windPdf * relativeForce,
            performance.windSpeeds)
    return loadFactor / costs.structureLifetime


def eco_kite(
    kite: KiteInputs,
    performance: PerformanceData,
    costs: KiteCosts,
    topology: Topology,
    business: BusinessInputs,
    availability: float = 1.0,
) -> Dict[str, Any]:
    """Calculate costs related to the kite subsystem.

    Args:
        kite (KiteInputs): Kite parameters.
        performance (PerformanceData): System performance data.
        costs (KiteCosts): Kite cost parameters.
        topology (Topology): System topology.
        business (BusinessInputs): Financial parameters (for the project
            lifetime used in the avionics/KCU replacement cap).
        availability (float): Fraction of the operating-wind time the
            system is flown [-], used by the reel-out-hour canopy
            replacement model. Defaults to 1.0.

    Returns:
        dict: The ``eco['kite']`` results subtree. The kite structure
        replacement (``structure.OPEX``) charges the structure CAPEX
        only; the avionics/KCU carries its own replacement OPEX, so the
        two are not bundled.
    """
    eco: Dict[str, Any] = {}

    # Structure
    if topology.wing == 'fixed':
        if costs.structureCostModel == KiteStructureCostModel.MASS_AREA:
            capex = (costs.priceStructuralMass * kite.mass +
                     costs.priceWettedArea * kite.flatArea)
        else:  # LAMINATE
            capex = ((1 + costs.manufacturingFactor) *
                     (costs.priceUniax * kite.uniaxMass +
                      costs.priceTriax * kite.triaxMass))
        replacementFrequency = kite.structureReplacementFrequency or 0.0
        eco['structure'] = {
            'CAPEX': capex,
            'OPEX': replacementFrequency * capex,
        }

    elif topology.wing == 'soft':
        capex = _soft_structure_capex(kite, costs)
        replacementFrequency = kite.structureReplacementFrequency
        if replacementFrequency is None:
            replacementFrequency = _soft_structure_replacement_frequency(
                kite, costs, performance, availability)
        eco['structure'] = {
            'CAPEX': capex,
            'OPEX': replacementFrequency * capex,
            'f_repl': replacementFrequency,
        }

    # Onboard generators and batteries
    if topology.power == 'FG':
        # Fly-gen: the onboard generator is sized by the rated power
        eco['obGen'] = {
            'CAPEX': (costs.onboardGeneratorPricePower *
                      performance.ratedPower / W_PER_KW),
            'OPEX': 0,
        }
    elif topology.power == 'GG':
        eco['obGen'] = {
            'CAPEX': (costs.onboardGeneratorPricePower *
                      kite.onboardGeneratorPower / W_PER_KW),
            'OPEX': 0,
        }
        eco['obBatt'] = {
            'CAPEX': (costs.onboardBatteryPriceEnergy *
                      kite.onboardBatteryCapacity),
            'OPEX': 0,
        }

    # Avionics / KCU: replacement OPEX = f_repl * CAPEX, with no
    # replacement charged for a life beyond the project.
    avionicsCapex = _avionics_capex(kite, costs)
    avionicsLife = costs.avionicsLifetime
    if (avionicsLife is not None and 0 < avionicsLife and
            avionicsLife <= business.nYears):
        avionicsReplacement = 1.0 / avionicsLife
    else:
        avionicsReplacement = 0.0
    eco['avionics'] = {
        'CAPEX': avionicsCapex,
        'OPEX': avionicsReplacement * avionicsCapex,
    }

    # Airborne sensor suite, itemised separately from the KCU. Same
    # replacement convention as the avionics.
    sensorCapex = costs.sensorCost or 0.0
    sensorLife = costs.sensorLifetime
    if (sensorLife is not None and 0 < sensorLife and
            sensorLife <= business.nYears):
        sensorReplacement = 1.0 / sensorLife
    else:
        sensorReplacement = 0.0
    eco['sensor'] = {
        'CAPEX': sensorCapex,
        'OPEX': sensorReplacement * sensorCapex,
    }

    return eco

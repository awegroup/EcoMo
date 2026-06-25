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
from ..eco_inputs import KiteInputs, PerformanceData, Topology


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
        # Two-term scaling model (soft-wing LEI):
        #   C_kite(S) = C_mat_ref * (S / S_ref)^b_mat + C_lab * S
        # TODO: C_mat_ref and S_ref are anchored to the 60 m2 LEI reference
        #       kite. Two anchor scenarios exist:
        #         Conservative (as-built prototype): C_mat_ref = 4146 EUR
        #         Lean AWE-optimised:                C_mat_ref = 3220 EUR
        #       Default uses the lean anchor (Scenario B from the cost model
        #       derivation). Update if more representative cost data appear.
        # TODO: b_mat = 1.14 is the cost-weighted component average (thesis
        #       Table 4.3). Uncertainty band: b=1.00 (pure area) to b=1.22
        #       (mass model). Sensitivity is significant for areas >100 m2.
        # TODO: C_lab = 13.0 EUR/m2 embeds sewing cost 1.30 EUR/m at 10 m/m2
        #       seam density. The 1.30 EUR/m rate is specific to the source
        #       labour rate (~7 EUR/h). For a different manufacturing context
        #       update via C_lab = (p_labour / v_sew) * seam_density, where
        #       v_sew = 5.4 m/h is empirically constant across architectures.
        return (costs.materialCostRef *
                (kite.flatArea / costs.referenceArea) **
                costs.materialScalingExponent +
                costs.labourCostCoefficient * kite.flatArea)
    # Fallback: flat-price model (Joshi & Trevisi 2024)
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
        # Scaled KCU model: fixed electronics + variable actuator portion
        #   C_avionics(S) = C_fixed + C_var_ref * (S / S_ref)^beta
        # TODO: C_fixed = 2400 EUR (30% of an 8000 EUR reference total) and
        #       C_var_ref = 5600 EUR (70%). Split from the Braun (2015) KCU
        #       product tree (~30% sensors/computing, ~70% drivetrains and
        #       mechanical structure). Verify with supplier quotes; likely
        #       low for prototype hardware (IEA Task 48: 15-30 kEUR).
        # TODO: beta = 1.0 (linear with area) assumes actuator force scales
        #       with tether force, hence with area. Range: beta=0.5 (Grete
        #       2014, avionics-dominated) to beta=1.5 (torque ~ force x lever
        #       arm). Run a sensitivity analysis for areas >100 m2.
        return (costs.avionicsCostFixed +
                costs.avionicsCostVarRef *
                (kite.flatArea / costs.avionicsReferenceArea) **
                costs.avionicsScalingExponent)
    # Fallback: fixed cost (Joshi & Trevisi 2024)
    return costs.avionicsCost


def _soft_structure_replacement_frequency(
    kite: KiteInputs,
    costs: KiteCosts,
    performance: PerformanceData,
) -> float:
    """Estimate the soft-wing structure replacement frequency.

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
    relativeForce = (performance.tetherForce /
                     np.max(performance.tetherForce))
    if (performance.reelOutTimeFraction is not None and
            np.any(performance.reelOutTimeFraction > 0)):
        # Time-weighted loading factor for pumping kites (corrected
        # formula): only the reel-out (traction) phase loads the kite.
        # The unweighted formula of Joshi & Trevisi (2024, Eq. 4)
        # overestimates the replacement frequency by ~38% for pumping
        # kites (unweighted LF = 0.355 vs time-weighted LF = 0.219).
        # TODO: reel-in phase loading is assumed negligible (~10-20% of
        #       the maximum). Valid for typical soft-wing pumping kites;
        #       revisit if the retraction strategy changes significantly.
        loadFactor = np.trapezoid(
            performance.windPdf * relativeForce *
            performance.reelOutTimeFraction,
            performance.windSpeeds)
    else:
        # Fallback: unweighted loading factor (Joshi & Trevisi 2024,
        # Eq. 4). This overestimates the replacement frequency by ~38%
        # for pumping kites because it ignores the near-zero loading of
        # the reel-in phase.
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
) -> Dict[str, Any]:
    """Calculate costs related to the kite subsystem.

    Args:
        kite (KiteInputs): Kite parameters.
        performance (PerformanceData): System performance data.
        costs (KiteCosts): Kite cost parameters.
        topology (Topology): System topology.

    Returns:
        dict: The ``eco['kite']`` results subtree.
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
                kite, costs, performance)
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

    # Avionics
    eco['avionics'] = {'CAPEX': _avionics_capex(kite, costs), 'OPEX': 0}

    return eco

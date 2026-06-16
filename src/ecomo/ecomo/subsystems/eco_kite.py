"""Kite subsystem economic calculations.

This module computes the capital expenditure (CAPEX) and operational
expenditure (OPEX) associated with the kite subsystem, including
structure, onboard generators, onboard batteries, and avionics.
"""

import numpy as np
from typing import Dict, Any

from ..constants import KiteStructureCostModel, W_PER_KW
from ..eco_costs import KiteCosts
from ..eco_inputs import KiteInputs, PerformanceData, Topology


def _soft_structure_replacement_frequency(
    kite: KiteInputs,
    costs: KiteCosts,
    performance: PerformanceData,
) -> float:
    """Estimate the soft-wing structure replacement frequency.

    The equivalent used lifetime per year follows from the loading
    factor (the wind-distribution-weighted tether force relative to the
    maximum force) divided by the structural lifetime at full loading.

    Args:
        kite (KiteInputs): Kite parameters.
        costs (KiteCosts): Kite cost parameters.
        performance (PerformanceData): System performance data.

    Returns:
        float: Replacement frequency [1/year].
    """
    loadFactor = np.trapezoid(
        performance.windPdf * performance.tetherForce /
        np.max(performance.tetherForce),
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
        capex = (costs.priceFabric + costs.priceBridle) * kite.flatArea
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
    eco['avionics'] = {'CAPEX': costs.avionicsCost, 'OPEX': 0}

    return eco

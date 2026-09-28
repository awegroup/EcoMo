"""Ground station subsystem economic calculations.

This module computes the capital and operational expenditures for the
ground station, including winch, drivetrain, generators, storage
systems, power converters, launch/land system, yaw system, and
control/communication unit.
"""

import numpy as np
from typing import Dict, Any, Tuple

from ..constants import (
    ComponentCostModel,
    DrivetrainType,
    HOURS_PER_YEAR,
    KWH_PER_MWH,
    SECONDS_PER_HOUR,
    StorageType,
    W_PER_KW,
)
from ..eco_costs import (
    GeneratorCosts,
    GroundStationCosts,
    StorageCosts,
    WinchCosts,
)
from ..eco_inputs import (
    BusinessInputs,
    GroundStationInputs,
    PerformanceData,
    StorageInputs,
    TetherInputs,
    Topology,
)


def _gg_storage_replacement_frequency(
    storage: StorageInputs,
    costs: StorageCosts,
    performance: PerformanceData,
) -> float:
    """Replacement frequency of a GG electrical storage bank.

    The number of charge/discharge cycles per year is estimated from
    the exchanged energy per pumping cycle and divided by the rated
    cycle life. Wind speeds where the cycle time is zero (system not
    operating) contribute no storage cycles.

    Args:
        storage (StorageInputs): Storage bank parameters.
        costs (StorageCosts): Storage cost parameters.
        performance (PerformanceData): System performance data.

    Returns:
        float: Replacement frequency [1/year].
    """
    exchangedEnergy = np.asarray(storage.exchangedEnergy, dtype=float)
    cycleTime = np.asarray(performance.cycleTime, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        energyTerm = exchangedEnergy / cycleTime * SECONDS_PER_HOUR
    energyTerm = np.nan_to_num(energyTerm, nan=0.0, posinf=0.0, neginf=0.0)

    return (HOURS_PER_YEAR *
            np.trapezoid(performance.windPdf * energyTerm,
                         performance.windSpeeds) /
            storage.ratedCapacity / costs.cycleLife)


def _fg_storage_replacement_frequency(
    costs: StorageCosts,
    performance: PerformanceData,
) -> float:
    """Replacement frequency of a FG electrical storage bank.

    For fly-gen systems the number of storage cycles is driven by the
    kite revolution frequency (tip-speed ratio over loop
    circumference).

    Args:
        costs (StorageCosts): Storage cost parameters.
        performance (PerformanceData): System performance data.

    Returns:
        float: Replacement frequency [1/year].
    """
    return (HOURS_PER_YEAR *
            np.trapezoid(performance.windPdf * performance.windSpeeds *
                         performance.tipSpeedRatio /
                         (2 * np.pi * performance.turningRadius),
                         performance.windSpeeds) /
            costs.cycleLife)


def _life_replacement_frequency(lifetime, n_years: int) -> float:
    """Replacement frequency [1/year] from a component service life.

    Convention for all lifetime-driven ground-station components (winch,
    gearbox, generator, power converter, launch & land, avionics/KCU,
    tether): the initial unit is annualised through the CRF as part of
    the ICC, and the replacement OPEX ``(1/life) * CAPEX`` charges only
    the re-buys during the project. A life of None, zero, or beyond the
    project means no replacement (the CRF annuity alone covers the
    capital), so no item is both fully annualised and re-charged
    annually without an actual replacement need.

    Args:
        lifetime: Component service life [years], or None.
        n_years (int): Project lifetime [years].

    Returns:
        float: Replacement frequency [1/year].
    """
    if lifetime is not None and 0 < lifetime <= n_years:
        return 1.0 / lifetime
    return 0.0


def _winch_cost(tether: TetherInputs, costs: WinchCosts,
                wallThickness: float) -> Tuple[float, float]:
    """Compute winch drum mass and CAPEX for a given wall thickness.

    Args:
        tether (TetherInputs): Tether parameters.
        costs (WinchCosts): Winch cost parameters.
        wallThickness (float): Winch drum wall thickness [m].

    Returns:
        tuple: ``(mass, capex)`` of the winch drum.
    """
    drumDiameter = costs.drumToTetherDiameterRatio * tether.diameter
    mass = (np.pi / 4 * (drumDiameter ** 2 -
                         (drumDiameter - 2 * wallThickness) ** 2) *
            tether.length * tether.diameter / (drumDiameter * np.pi) *
            costs.density *
            costs.safetyFactorDiameter * costs.safetyFactorLength)
    return mass, mass * costs.price


def _generator_costs(costs: GeneratorCosts,
                     sizingPower: float) -> Dict[str, Any]:
    """Compute the electric generator cost subtree.

    Args:
        costs (GeneratorCosts): Generator cost parameters.
        sizingPower (float): Power used to size the generator [W].

    Returns:
        dict: The ``eco['gStation']['gen']`` subtree.
    """
    if costs.costModel == ComponentCostModel.POWER_BASED:
        return {'CAPEX': costs.pricePower * sizingPower / W_PER_KW,
                'OPEX': 0}
    mass = costs.massSlope * sizingPower / W_PER_KW + costs.massOffset
    return {'m': mass, 'CAPEX': costs.priceMass * mass, 'OPEX': 0}


def _storage_costs(storage: StorageInputs, costs: StorageCosts,
                   autoFrequency) -> Dict[str, Any]:
    """Compute the cost subtree of an electrical storage bank.

    Args:
        storage (StorageInputs): Storage bank parameters.
        costs (StorageCosts): Storage cost parameters.
        autoFrequency (callable): Zero-argument function returning the
            auto-estimated replacement frequency when none is set.

    Returns:
        dict: The storage cost subtree (CAPEX/OPEX/f_repl).
    """
    capex = costs.priceEnergy * storage.ratedCapacity
    frequency = storage.replacementFrequency
    if frequency is None:
        frequency = autoFrequency()
    return {'CAPEX': capex, 'OPEX': frequency * capex, 'f_repl': frequency}


def _electrical_storage_costs(
    groundStation: GroundStationInputs,
    costs: GroundStationCosts,
    performance: PerformanceData,
    topology: Topology,
) -> Dict[str, Any]:
    """Compute the cost subtree of the selected electrical storage.

    Args:
        groundStation (GroundStationInputs): Ground station parameters.
        costs (GroundStationCosts): Ground station cost parameters.
        performance (PerformanceData): System performance data.
        topology (Topology): System topology.

    Returns:
        dict: Mapping of the selected storage key ('ultracap' or
        'batt') to its cost subtree.
    """
    if costs.electricalStorage == StorageType.ULTRACAPACITOR:
        key, storage = 'ultracap', groundStation.ultracapacitor
    else:
        key, storage = 'batt', groundStation.battery
    storageCosts = costs.storage(costs.electricalStorage)

    if topology.power == 'GG':
        def auto():
            return _gg_storage_replacement_frequency(
                storage, storageCosts, performance)
    else:
        def auto():
            return _fg_storage_replacement_frequency(
                storageCosts, performance)

    return {key: _storage_costs(storage, storageCosts, auto)}


def eco_gstation(
    tether: TetherInputs,
    groundStation: GroundStationInputs,
    performance: PerformanceData,
    costs: GroundStationCosts,
    tether_max_stress: float,
    business: BusinessInputs,
    topology: Topology,
    kite_flat_area: float = 0.0,
) -> Dict[str, Any]:
    """Calculate costs related to the ground station subsystem.

    Args:
        tether (TetherInputs): Tether parameters.
        groundStation (GroundStationInputs): Ground station parameters.
        performance (PerformanceData): System performance data.
        costs (GroundStationCosts): Ground station cost parameters.
        tether_max_stress (float): Maximum tether fibre stress [Pa],
            used to size the winch drum thickness.
        business (BusinessInputs): Financial parameters (for the
            project lifetime used in the launch & land replacement).
        topology (Topology): System topology.
        kite_flat_area (float): Flat wing area [m2], used to size the
            launch & land system when it is priced per area. Defaults to
            0.0 (no area-scaled contribution).

    Returns:
        dict: The ``eco['gStation']`` results subtree.
    """
    eco: Dict[str, Any] = {}
    drumDiameter = costs.winch.drumToTetherDiameterRatio * tether.diameter
    winchReplacement = _life_replacement_frequency(
        costs.winch.lifetime, business.nYears)

    if topology.power == 'GG':
        # Winch; the drum wall thickness is sized by the tether load
        wallThickness = (np.pi / 4 * tether_max_stress /
                         costs.winch.maxStress * tether.diameter)
        winchMass, winchCapex = _winch_cost(tether, costs.winch,
                                            wallThickness)
        eco['winch'] = {'D': drumDiameter, 't': wallThickness,
                        'm': winchMass, 'CAPEX': winchCapex,
                        'OPEX': winchReplacement * winchCapex}

        if costs.drivetrain == DrivetrainType.ELECTRIC:
            # Gearbox, sized by the peak mechanical power or torque
            gearboxReplacement = _life_replacement_frequency(
                costs.gearbox.lifetime, business.nYears)
            if costs.gearbox.costModel == ComponentCostModel.POWER_BASED:
                gearboxCapex = (costs.gearbox.pricePower *
                                performance.peakMechanicalPower / W_PER_KW)
                eco['gearbox'] = {
                    'CAPEX': gearboxCapex,
                    'OPEX': gearboxReplacement * gearboxCapex,
                }
            else:
                mass = (costs.gearbox.massCoefficient *
                        (np.max(performance.tetherForce) *
                         drumDiameter / 2 / W_PER_KW) **
                        costs.gearbox.massExponent)
                gearboxCapex = costs.gearbox.priceMass * mass
                eco['gearbox'] = {
                    'm': mass,
                    'CAPEX': gearboxCapex,
                    'OPEX': gearboxReplacement * gearboxCapex,
                }

            # Electric generator, sized by the peak mechanical power
            eco['gen'] = _generator_costs(
                costs.generator, performance.peakMechanicalPower)
            eco['gen']['OPEX'] = (
                _life_replacement_frequency(costs.generator.lifetime,
                                            business.nYears) *
                eco['gen']['CAPEX'])

            # Electrical storage
            eco.update(_electrical_storage_costs(
                groundStation, costs, performance, topology))

            # Power converters
            powerConvCapex = (costs.powerConverterPricePower *
                              (performance.ratedPower +
                               performance.peakMechanicalPower) / W_PER_KW)
            eco['powerConv'] = {
                'CAPEX': powerConvCapex,
                'OPEX': (_life_replacement_frequency(
                    costs.powerConverterLifetime, business.nYears) *
                    powerConvCapex),
            }

        elif costs.drivetrain == DrivetrainType.HYDRAULIC:
            # Pump-motor, sized by the peak mechanical power
            eco['pumpMotor'] = {
                'CAPEX': (costs.pumpMotorPricePower *
                          performance.peakMechanicalPower / W_PER_KW),
                'OPEX': (groundStation.pumpMotorReplacementFrequency *
                         costs.pumpMotorMaintenancePricePower *
                         performance.peakMechanicalPower / W_PER_KW),
            }

            # Hydropneumatic accumulator bank
            accumulator = groundStation.hydraulicAccumulator
            eco['hydAccum'] = {
                'CAPEX': (costs.hydraulicAccumulatorPriceEnergy *
                          accumulator.ratedCapacity / KWH_PER_MWH),
                'OPEX': (accumulator.replacementFrequency *
                         costs.hydraulicAccumulatorMaintenancePriceEnergy *
                         accumulator.exchangedEnergy / KWH_PER_MWH),
            }

            # Hydraulic motor, sized by the rated electrical power
            eco['hydMotor'] = {
                'CAPEX': (costs.hydraulicMotorPricePower *
                          performance.ratedPower / W_PER_KW),
                'OPEX': (groundStation.hydraulicMotorReplacementFrequency *
                         costs.hydraulicMotorMaintenancePricePower *
                         performance.ratedPower / W_PER_KW),
            }

            # Electric generator, sized by the rated electrical power
            eco['gen'] = _generator_costs(
                costs.generator, performance.ratedPower)
            eco['gen']['OPEX'] = (
                _life_replacement_frequency(costs.generator.lifetime,
                                            business.nYears) *
                eco['gen']['CAPEX'])

    elif topology.power == 'FG':
        # Winch; the drum wall thickness equals the tether diameter
        winchMass, winchCapex = _winch_cost(tether, costs.winch,
                                            tether.diameter)
        eco['winch'] = {'D': drumDiameter, 't': tether.diameter,
                        'm': winchMass, 'CAPEX': winchCapex,
                        'OPEX': winchReplacement * winchCapex}

        # Electrical storage
        eco.update(_electrical_storage_costs(
            groundStation, costs, performance, topology))

        # Power converters
        powerConvCapex = (2 * costs.powerConverterPricePower *
                          performance.ratedPower / W_PER_KW)
        eco['powerConv'] = {
            'CAPEX': powerConvCapex,
            'OPEX': (_life_replacement_frequency(
                costs.powerConverterLifetime, business.nYears) *
                powerConvCapex),
        }

    # Launch & land system (both FG and GG): area-scaled CAPEX when a
    # per-area price is set, otherwise a fixed cost, with a replacement
    # OPEX when its life is shorter than the project.
    if costs.launchLandPriceArea is not None:
        launchLandCapex = costs.launchLandPriceArea * kite_flat_area
    else:
        launchLandCapex = costs.launchLandCost or 0.0
    launchLandReplacement = _life_replacement_frequency(
        costs.launchLandLifetime, business.nYears)
    eco['lls'] = {
        'CAPEX': launchLandCapex,
        'OPEX': launchLandReplacement * launchLandCapex,
    }
    eco['yaw'] = {'CAPEX': 0, 'OPEX': 0}
    eco['controlStation'] = {'CAPEX': 0, 'OPEX': 0}

    return eco

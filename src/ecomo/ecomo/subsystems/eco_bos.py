"""Balance of System (BoS) subsystem economic calculations.

This module computes the capital and operational expenditures for the
BoS subsystem, including site preparation, foundation, installation,
O&M, decommissioning, the operator/maintenance crew labour and the
recurring consumables bundle.

Three clearly separated annual operating terms are produced, so each is
traceable on its own (rather than everything summed inside ``BoS.OM``):

- ``BoS.OM`` -- the per-kW O&M overhead
  (``operations_maintenance.price_power * ratedPowerKw``). This is the
  size-dependent term whose cost scales with the rated power (the
  general non-labour upkeep carried over from the earlier MATLAB
  AWE-Eco implementation); it is *not* land/insurance or labour.

- ``BoS.labour`` -- the explicit operator and maintenance crew, the
  dominant operating cost for small, manually-supervised AWE prototypes:
    * operation (base-crew) labour, per operating day and
      size-independent, ``C_op = (1 - automation) * w * N_op * h_op``
      (rig-up, launch, monitor, land, pack-down; automatable). The plant
      is manned on ``N_op = f_wind * 365`` operating days per year (the
      windy fraction of the year);
    * maintenance labour, per flight hour,
      ``C_maint = w * h_maint_per_flight_h * annual_flight_hours``
      (routine inspection/upkeep of the wearing airborne parts, which
      accrues with time actually flown rather than with the calendar).

- ``BoS.consumables`` -- the recurring consumable bundle (AWT, kite bag,
  weak link, sensors) replaced yearly, a fixed annual EUR amount scaled
  by a maturity factor (``matures down`` as the system matures).

The three do not double-count: the per-kW O&M overhead covers only the
size-dependent standing upkeep, the labour group covers the crew, and
the consumables cover the yearly replaced parts.
"""

from typing import Dict, Any, Optional

from ..constants import W_PER_KW
from ..eco_costs import BalanceOfSystemCosts
from ..eco_hours import annual_flight_hours, annual_operating_days
from ..eco_inputs import OperationsInputs, PerformanceData


def _operation_maintenance_labour(
    operations: OperationsInputs,
    performance: PerformanceData,
    availability: float = 1.0,
) -> Dict[str, float]:
    """Operation and maintenance labour OPEX [EUR/year].

    Operation (base-crew) labour follows an operating-day model: the
    plant is manned on ``N_op = f_wind * 365`` operating days per year,
    each carrying a fixed (size-independent) operating allowance;
    automation scales only this term. Maintenance labour is charged per
    flight hour, so it accrues with the time actually flown.

    Args:
        operations (OperationsInputs): Operations/labour parameters.
        performance (PerformanceData): System performance data.
        availability (float): Fraction of the operating-wind time flown
            [-], scaling the annual flight hours the maintenance labour
            is charged on.

    Returns:
        dict: ``operation`` and ``maintenance`` labour OPEX with the
        annual ``operating_days`` and ``flight_hours`` counts they are
        built on.
    """
    # Operating days = windy fraction of the year (availability-independent);
    # the base crew is a standing cost tied to the operating calendar.
    operatingDays = annual_operating_days(performance)
    # Flight hours scale with availability; maintenance accrues with them.
    flightHours = annual_flight_hours(performance, availability)

    operationOpex = ((1.0 - operations.automation) *
                     operations.labourPrice * operatingDays *
                     operations.operatingHoursPerDay)
    maintenanceOpex = (operations.labourPrice *
                       operations.maintenanceHoursPerFlightHour * flightHours)
    return {
        'operation': operationOpex,
        'maintenance': maintenanceOpex,
        'operating_days': operatingDays,
        'flight_hours': flightHours,
    }


def eco_bos(
    performance: PerformanceData,
    costs: BalanceOfSystemCosts,
    operations: Optional[OperationsInputs] = None,
    availability: float = 1.0,
) -> Dict[str, Any]:
    """Calculate costs related to the Balance of System subsystem.

    Args:
        performance (PerformanceData): System performance data (for the
            rated and peak mechanical power).
        costs (BalanceOfSystemCosts): BoS cost parameters.
        operations (OperationsInputs): Operations/labour parameters, or
            None to charge no explicit crew labour (only the per-kW O&M
            overhead and the recurring consumables are then charged).
        availability (float): Fraction of the operating-wind time flown
            [-], used by the per-flight-hour maintenance labour.
            Defaults to 1.0.

    Returns:
        dict: The ``eco['BoS']`` results subtree, with the three separate
        operating leaves ``OM`` (per-kW overhead), ``labour`` (crew) and
        ``consumables`` (recurring bundle).
    """
    ratedPowerKw = performance.ratedPower / W_PER_KW

    # The foundation is sized by the peak mechanical load, not the rated
    # electrical power (Joshi & Trevisi 2024, Eq. 38: C_found = p_found *
    # P_peak). For soft-wing pumping kites P_peak is well above P_rated
    # (~2.9x in the reference case); use the actual peak power when
    # available, otherwise fall back to the rated power.
    if performance.peakMechanicalPower is not None:
        peakPowerKw = performance.peakMechanicalPower / W_PER_KW
    else:
        peakPowerKw = ratedPowerKw

    installCapex = costs.installationPricePower * ratedPowerKw

    # O&M overhead (per-kW, size-dependent upkeep; no labour, no consumables)
    overheadOpex = costs.operationsMaintenancePricePower * ratedPowerKw

    result: Dict[str, Any] = {
        'sitePrep': {'CAPEX': costs.sitePreparationPricePower * ratedPowerKw},
        'found': {'CAPEX': costs.foundationPricePower * peakPowerKw},
        'install': {'CAPEX': installCapex},
        'OM': {'overhead_opex': overheadOpex, 'OPEX': overheadOpex},
        'decomm': {'CAPEX': (costs.decommissioningInstallationFraction *
                             installCapex)},
    }

    # Crew labour: its own leaf group, separate from BoS.OM, so it is
    # traceable on its own (the CAPEX/OPEX aggregation only sums leaves
    # named exactly 'CAPEX'/'OPEX', so the diagnostic keys are ignored).
    if operations is not None:
        labour = _operation_maintenance_labour(
            operations, performance, availability)
        result['labour'] = {
            'operation_opex': labour['operation'],
            'maintenance_opex': labour['maintenance'],
            'operating_days': labour['operating_days'],
            'flight_hours': labour['flight_hours'],
            'OPEX': labour['operation'] + labour['maintenance'],
        }

    # Recurring consumables (AWT, kite bag, weak link, sensors) replaced
    # yearly. A maturity factor scales the base annual cost down as the
    # system matures (fewer replacements). Its own leaf, distinct from
    # BoS.OM and the labour group.
    consumablesOpex = costs.consumablesEurPerYear * costs.consumablesMaturity
    result['consumables'] = {
        'annual_eur': costs.consumablesEurPerYear,
        'maturity': costs.consumablesMaturity,
        'OPEX': consumablesOpex,
    }

    return result

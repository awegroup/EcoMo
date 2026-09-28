"""Balance of System (BoS) subsystem economic calculations.

Computes the BoS CAPEX (site preparation, foundation, installation,
decommissioning) and the two separate annual operating leaves: ``BoS.OM``
(the per-kW O&M overhead) and ``BoS.labour`` (the operator and maintenance
crew). See reports/COST_MODEL_REFERENCE.md for the labour model.
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
            overhead is then charged).
        availability (float): Fraction of the operating-wind time flown
            [-], used by the per-flight-hour maintenance labour.
            Defaults to 1.0.

    Returns:
        dict: The ``eco['BoS']`` results subtree, with the two separate
        operating leaves ``OM`` (per-kW overhead) and ``labour`` (crew).
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

    # O&M overhead (per-kW, size-dependent upkeep; no labour)
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

    return result

"""Balance of System (BoS) subsystem economic calculations.

This module computes the capital and operational expenditures for the
BoS subsystem, including site preparation, foundation, installation,
O&M, and decommissioning.

The operations & maintenance (``OM``) OPEX collects the annual operating
cost of the plant. It is the sum of:

- the per-kW BoS O&M overhead (land lease, insurance, general upkeep;
  the term carried over from the earlier MATLAB AWE-Eco implementation);
- the explicit operator/maintenance labour, when an operations block is
  configured. The labour is the dominant operating cost for small,
  manually-supervised AWE prototypes and follows an operating-day model.
  The plant is manned on ``N_op = f_wind * 365`` operating days per year
  (the windy fraction of the year); each operating day carries a fixed
  operating and maintenance allowance, size-independent:
    * operation labour  C_op    = (1 - automation) * w_lab * N_op * h_op
      (rig-up, launch, monitor, land, pack-down; automatable);
    * maintenance labour C_maint = w_lab * N_op * h_m
      (routine inspection/upkeep of the wearing airborne parts).

The per-day hours are size-independent on purpose: one crew runs one
system regardless of wing area, so a fixed crew is amortised over a
larger, more energetic system as it scales (economy of scale in labour).

The labour lives here under ``BoS.OM`` rather than in a separate
operations subsystem: it is an operating expense of the same nature as
the per-kW O&M overhead. The two do not double-count: the per-kW O&M
overhead covers only non-labour standing costs (land lease, insurance,
general upkeep), while the explicit labour covers the operator and
maintenance crew -- which is essential to model for soft-wing systems,
still manually operated and needing frequent replacement of the soft
structure.
"""

from typing import Dict, Any, Optional

from ..constants import W_PER_KW
from ..eco_costs import BalanceOfSystemCosts
from ..eco_hours import annual_operating_days
from ..eco_inputs import OperationsInputs, PerformanceData


def _operation_maintenance_labour(
    operations: OperationsInputs,
    performance: PerformanceData,
) -> Dict[str, float]:
    """Operation and maintenance labour OPEX [EUR/year].

    Operating-day model: the plant is manned on ``N_op = f_wind * 365``
    operating days per year, each carrying a fixed (size-independent)
    operating and maintenance allowance. Automation scales only the
    operating term.

    Args:
        operations (OperationsInputs): Operations/labour parameters.
        performance (PerformanceData): System performance data.

    Returns:
        dict: ``operation`` and ``maintenance`` labour OPEX and the
        annual ``operating_days`` count they are built on.
    """
    # Operating days = windy fraction of the year (availability-independent);
    # the crew is a standing cost tied to the operating calendar.
    operatingDays = annual_operating_days(performance)

    operationOpex = ((1.0 - operations.automation) *
                     operations.labourPrice * operatingDays *
                     operations.operatingHoursPerDay)
    maintenanceOpex = (operations.labourPrice * operatingDays *
                       operations.maintenanceHoursPerDay)
    return {
        'operation': operationOpex,
        'maintenance': maintenanceOpex,
        'operating_days': operatingDays,
    }


def eco_bos(
    performance: PerformanceData,
    costs: BalanceOfSystemCosts,
    operations: Optional[OperationsInputs] = None,
) -> Dict[str, Any]:
    """Calculate costs related to the Balance of System subsystem.

    Args:
        performance (PerformanceData): System performance data (for the
            rated and peak mechanical power).
        costs (BalanceOfSystemCosts): BoS cost parameters.
        operations (OperationsInputs): Operations/labour parameters, or
            None to charge only the per-kW O&M overhead (no explicit
            labour).

    Returns:
        dict: The ``eco['BoS']`` results subtree. ``OM.OPEX`` is the sum
        of the per-kW O&M overhead and the operator/maintenance labour;
        the individual labour components are kept as diagnostics.
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

    # O&M overhead (per-kW, non-labour: land lease, insurance, upkeep)
    overheadOpex = costs.operationsMaintenancePricePower * ratedPowerKw

    om: Dict[str, Any] = {'overhead_opex': overheadOpex}
    omOpex = overheadOpex
    if operations is not None:
        labour = _operation_maintenance_labour(operations, performance)
        # Diagnostics (ignored by the CAPEX/OPEX aggregation, which only
        # sums leaves named exactly 'CAPEX'/'OPEX')
        om['operation_labour_opex'] = labour['operation']
        om['maintenance_labour_opex'] = labour['maintenance']
        om['operating_days'] = labour['operating_days']
        omOpex += labour['operation'] + labour['maintenance']
    om['OPEX'] = omOpex

    return {
        'sitePrep': {'CAPEX': costs.sitePreparationPricePower * ratedPowerKw},
        'found': {'CAPEX': costs.foundationPricePower * peakPowerKw},
        'install': {'CAPEX': installCapex},
        'OM': om,
        'decomm': {'CAPEX': (costs.decommissioningInstallationFraction *
                             installCapex)},
    }

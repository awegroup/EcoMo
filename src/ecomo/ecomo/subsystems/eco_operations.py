"""Operations subsystem economic calculations.

Computes the explicit labour OPEX, which is the dominant operating cost
for small, manually-supervised AWE systems and is not captured by the
per-kW balance-of-system O&M term. The labour is split into three terms:

- Supervision (fixed): size-independent monitoring/piloting presence.
- Launch/recovery: per-operation hands-on labour, scaled down by the
  launch/recovery automation fraction.
- Maintenance: labour proportional to the annual flight hours.
"""

from typing import Dict, Any

from ..eco_hours import annual_flight_hours as _annual_flight_hours
from ..eco_inputs import OperationsInputs, PerformanceData

WEEKS_PER_YEAR = 52


def annual_flight_hours(operations: OperationsInputs,
                        performance: PerformanceData) -> float:
    """Estimate the annual operating (flight) hours.

    Thin wrapper around :func:`ecomo.ecomo.eco_hours.annual_flight_hours`
    that supplies the availability from the operations parameters.

    Note:
        ``availability`` scales the flight hours (hence the labour and
        load-driven O&M) and, in :func:`eco_compute_metrics`, the net
        AEP, so downtime raises cost and lowers energy consistently.

    Args:
        operations (OperationsInputs): Operations parameters (for the
            availability).
        performance (PerformanceData): System performance data.

    Returns:
        float: Annual flight hours [h/year].
    """
    return _annual_flight_hours(performance, operations.availability)


def eco_operations(
    operations: OperationsInputs,
    performance: PerformanceData,
) -> Dict[str, Any]:
    """Calculate the labour OPEX (supervision, launch/recovery, maintenance).

    - Supervision is a fixed, size-independent cost
      ``w_lab * H_sup`` (``H_sup = operator_hours_per_week * 52``).
    - Launch/recovery is a per-operation cost
      ``(1 - alpha_auto) * N_op * h_LR * w_lab`` (added only when
      configured), distinct from the supervision presence so there is
      no double-count. ``N_op`` (operations per year) is an operating-
      pattern *assumption*, not a derived quantity -- the launch/recovery
      frequency needs a wind time series plus a landing policy, which
      this distribution-based model does not have -- so it is treated as
      an uncertain/swept input (see the TODO in the settings YAML).
    - Maintenance scales with the annual flight hours
      ``r_m * H_flight * w_lab``.

    Args:
        operations (OperationsInputs): Operations parameters.
        performance (PerformanceData): System performance data.

    Returns:
        dict: The ``eco['operations']`` results subtree.
    """
    flightHours = annual_flight_hours(operations, performance)

    supervisionOpex = (operations.operatorHoursPerWeek * WEEKS_PER_YEAR *
                       operations.labourPrice)
    maintenanceOpex = (operations.maintenanceHoursPerFlightHour *
                       flightHours * operations.labourPrice)

    result: Dict[str, Any] = {
        'supervision': {'OPEX': supervisionOpex},
        'maintenance': {'OPEX': maintenanceOpex},
        # Diagnostics (ignored by the metrics CAPEX/OPEX aggregation)
        'flight_hours': flightHours,
    }

    # Launch/recovery labour: per-operation, hands-on, reduced by the
    # automation fraction. Only charged when both a count and an
    # hours-per-operation are configured.
    if (operations.operationsPerYear and
            operations.launchRecoveryHoursPerOperation):
        result['launch_recovery'] = {
            'OPEX': ((1.0 - operations.launchAutomation) *
                     operations.operationsPerYear *
                     operations.launchRecoveryHoursPerOperation *
                     operations.labourPrice),
        }

    return result

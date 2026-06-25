"""Balance of System (BoS) subsystem economic calculations.

This module computes the capital and operational expenditures for the
BoS subsystem, including site preparation, foundation, installation,
O&M, and decommissioning.
"""

from typing import Dict, Any

from ..constants import W_PER_KW
from ..eco_costs import BalanceOfSystemCosts
from ..eco_inputs import PerformanceData


def eco_bos(
    performance: PerformanceData,
    costs: BalanceOfSystemCosts,
) -> Dict[str, Any]:
    """Calculate costs related to the Balance of System subsystem.

    Args:
        performance (PerformanceData): System performance data (for the
            rated and peak mechanical power).
        costs (BalanceOfSystemCosts): BoS cost parameters.

    Returns:
        dict: The ``eco['BoS']`` results subtree.
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
    return {
        'sitePrep': {'CAPEX': costs.sitePreparationPricePower * ratedPowerKw},
        'found': {'CAPEX': costs.foundationPricePower * peakPowerKw},
        'install': {'CAPEX': installCapex},
        'OM': {'OPEX': costs.operationsMaintenancePricePower * ratedPowerKw},
        'decomm': {'CAPEX': (costs.decommissioningInstallationFraction *
                             installCapex)},
    }

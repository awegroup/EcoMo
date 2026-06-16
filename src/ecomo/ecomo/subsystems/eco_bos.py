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
            rated power).
        costs (BalanceOfSystemCosts): BoS cost parameters.

    Returns:
        dict: The ``eco['BoS']`` results subtree.
    """
    ratedPowerKw = performance.ratedPower / W_PER_KW

    installCapex = costs.installationPricePower * ratedPowerKw
    return {
        'sitePrep': {'CAPEX': costs.sitePreparationPricePower * ratedPowerKw},
        'found': {'CAPEX': costs.foundationPricePower * ratedPowerKw},
        'install': {'CAPEX': installCapex},
        'OM': {'OPEX': costs.operationsMaintenancePricePower * ratedPowerKw},
        'decomm': {'CAPEX': (costs.decommissioningInstallationFraction *
                             installCapex)},
    }

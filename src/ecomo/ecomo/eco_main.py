"""Orchestration of the ECOMo simulation.

Runs the subsystem cost modules (kite, tether, ground station, BoS,
BoP) and the metrics calculation, assembling their results into the
``eco`` results dictionary.
"""

from typing import Dict, Any

from .eco_costs import EcoCosts
from .eco_inputs import EcoInputs
from .subsystems import (
    eco_kite,
    eco_tether,
    eco_gstation,
    eco_bos,
    eco_bop,
)
from .eco_metrics import eco_compute_metrics


def eco_main(inputs: EcoInputs, costs: EcoCosts) -> Dict[str, Any]:
    """Run the ECOMo simulation for one system configuration.

    Executes all subsystem cost modules and computes the economic
    metrics.

    Args:
        inputs (EcoInputs): The complete input set.
        costs (EcoCosts): The complete cost parameter set.

    Returns:
        dict: The ``eco`` results structure, with one subtree per
        subsystem plus a ``metrics`` subtree.
    """
    eco: Dict[str, Any] = {
        'kite': eco_kite(inputs.kite, inputs.performance,
                         costs.kite, inputs.topology),
        'tether': eco_tether(inputs.tether, inputs.performance,
                             costs.tether, inputs.business, inputs.topology),
        'gStation': eco_gstation(inputs.tether, inputs.groundStation,
                                 inputs.performance, costs.groundStation,
                                 costs.tether.maxStress, inputs.topology),
        'BoS': eco_bos(inputs.performance, costs.balanceOfSystem),
        'BoP': eco_bop(),
    }

    eco['metrics'] = eco_compute_metrics(eco, inputs.business,
                                         inputs.performance, costs.market)

    return eco

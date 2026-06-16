"""Load system performance data from a performance YAML file.

Reads a ``system_performance_*.yml`` file holding the wind-dependent
performance-model outputs the economic model needs in standalone mode
(in AWESPA-connected mode the same data come from ``power_curves.yml``
and ``aep_results.yml`` instead). Field names are aligned with the
awesIO ``power_curves_schema.yml`` where an equivalent exists.

Optional fields fall back to the assumptions in Joshi & Trevisi (2024):
the peak mechanical power defaults to 2.5x the rated electrical power
(section 4.2.1), the pumping cycle time to 60 s (section 3) and the
storage exchanged energy to half the rated storage capacity (Eq. 26).
A warning is emitted whenever such a fallback is used.
"""

import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml

# Peak mechanical reel-out power to rated electrical power ratio
# (Joshi & Trevisi 2024, section 4.2.1, ballpark value for GG systems)
PEAK_TO_RATED_POWER = 2.5

# Default pumping cycle time when absent for a GG system
# (Joshi & Trevisi 2024, section 3, assumed value if none available)
DEFAULT_CYCLE_TIME = 60.0


def eco_load_performance(performance_path: Path, power: str) -> Dict[str, Any]:
    """Load performance-model outputs from a YAML file.

    Args:
        performance_path (Path): Path to a ``system_performance_*.yml``
            file.
        power (str): Power generation type, 'GG' or 'FG', used to decide
            which optional fields warrant a warning when absent.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a required field is missing.
        ValueError: If a per-wind-speed array does not match the wind
            speed grid.

    Returns:
        dict: Performance data with keys ``'windRange'`` [m/s],
        ``'peAvg'`` [W], ``'peRated'`` [W], ``'pmPeak'`` [W or None],
        ``'dtCycle'`` [s], ``'tetherForce'`` [N or None],
        ``'exchangedEnergy'`` (dict per storage name, kWh),
        ``'tipSpeedRatio'`` and ``'turningRadius'`` (FG only, may be
        None).
    """
    performancePath = Path(performance_path)
    if not performancePath.exists():
        raise FileNotFoundError(
            f"Performance file not found: {performancePath}")

    with open(performancePath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    windRange = np.asarray(data['reference_wind_speeds'], dtype=float)

    def as_wind_array(value, name):
        """Convert a scalar or list to the wind grid, checking length."""
        if np.isscalar(value):
            return float(value)
        array = np.asarray(value, dtype=float)
        if len(array) != len(windRange):
            raise ValueError(
                f"'{name}' has {len(array)} entries but "
                f"'reference_wind_speeds' has {len(windRange)} "
                f"in {performancePath.name}")
        return array

    peRated = float(data['rated_electrical_power'])

    pmPeak = data.get('peak_mechanical_power')
    if pmPeak is not None:
        pmPeak = float(pmPeak)

    exchangedEnergy = {
        name: as_wind_array(value, f'storage_exchanged_energy.{name}')
        for name, value in
        (data.get('storage_exchanged_energy') or {}).items()
    }

    # Cycle time: needed by GG bending-life and storage-cycle estimates;
    # falls back to the default 60 s if absent for a GG system
    if 'cycle_time' in data:
        dtCycle = as_wind_array(data['cycle_time'], 'cycle_time')
    elif power == 'GG':
        warnings.warn(
            "'cycle_time' not found in performance file; using the "
            f"default pumping cycle time of {DEFAULT_CYCLE_TIME:.0f} s "
            "(Joshi & Trevisi 2024, section 3). Set 'cycle_time' in the "
            "performance YAML to use the exact value.",
            UserWarning, stacklevel=2,
        )
        dtCycle = DEFAULT_CYCLE_TIME
    else:
        dtCycle = None

    # Tether force: absent means it is set to zero downstream, which
    # disables the tether and soft-kite replacement estimates
    if 'tether_force' in data:
        tetherForce = as_wind_array(data['tether_force'], 'tether_force')
    else:
        warnings.warn(
            "'tether_force' not found in performance file; tether force "
            "set to zero at all wind speeds. Tether and soft-kite "
            "replacement costs will not be estimated. Set 'tether_force' "
            "in the performance YAML to enable these calculations.",
            UserWarning, stacklevel=2,
        )
        tetherForce = None

    # Tip-speed ratio: needed by the FG storage-replacement estimate
    tipSpeedRatio = data.get('tip_speed_ratio')
    if tipSpeedRatio is None and power == 'FG':
        warnings.warn(
            "'tip_speed_ratio' not found in performance file; it is "
            "needed for the fly-gen storage replacement estimate. Set "
            "'tip_speed_ratio' in the performance YAML.",
            UserWarning, stacklevel=2,
        )

    return {
        'windRange': windRange,
        'peAvg': as_wind_array(data['average_cycle_power'],
                               'average_cycle_power'),
        'peRated': peRated,
        'pmPeak': pmPeak,
        'dtCycle': dtCycle,
        'tetherForce': tetherForce,
        'exchangedEnergy': exchangedEnergy,
        'tipSpeedRatio': tipSpeedRatio,
        'turningRadius': data.get('turning_radius'),
    }


def peak_mechanical_power_fallback(peRated: float) -> float:
    """Estimate the peak mechanical power from the rated power.

    Args:
        peRated (float): Rated electrical power [W].

    Returns:
        float: Estimated peak mechanical reel-out power [W].
    """
    warnings.warn(
        "'peak_mechanical_power' not available; using the ballpark "
        f"estimate {PEAK_TO_RATED_POWER} x rated electrical power "
        "(Joshi & Trevisi 2024, section 4.2.1). Provide "
        "'peak_mechanical_power' to use the exact value.",
        UserWarning, stacklevel=2,
    )
    return PEAK_TO_RATED_POWER * peRated

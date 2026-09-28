"""Annual operating-hour estimates derived from the performance data.

These helpers turn the wind-speed performance curves into annual hour and
cycle counts, shared by the operations (labour), kite (canopy life) and
tether (operational life) cost models so they all use one consistent
operating definition.

The operating band is the share of the year the wind sits where the system
produces power (non-zero average power), weighted by the wind distribution
and scaled by the availability (the fraction of that time actually flown).

Note:
    ``availability`` scales the hour and cycle counts here (hence the labour
    and the load-driven replacement costs) and, in ``eco_compute_metrics``,
    the net AEP as well, so downtime reduces both cost and delivered energy
    consistently.
"""

import numpy as np
from typing import Optional

from .constants import HOURS_PER_YEAR, SECONDS_PER_HOUR
from .eco_inputs import PerformanceData

DAYS_PER_YEAR = 365.0


def _operating_mask(performance: PerformanceData) -> np.ndarray:
    """Boolean mask of wind speeds where the system produces power."""
    return np.asarray(performance.averagePower, dtype=float) > 0


def wind_operability_fraction(performance: PerformanceData) -> float:
    """Fraction of the year the wind is in the operating band [-].

    The share of the year the wind sits where the system produces power
    (non-zero average power), weighted by the wind distribution. Unlike
    :func:`annual_flight_hours` it carries no availability factor, so it
    is a pure resource quantity.

    Args:
        performance (PerformanceData): System performance data.

    Returns:
        float: Wind-operability fraction ``f_wind`` in [0, 1].
    """
    return float(np.trapezoid(
        performance.windPdf * _operating_mask(performance),
        performance.windSpeeds))


def annual_operating_days(performance: PerformanceData) -> float:
    """Annual operating days [days/year].

    The operating days are the windy days of the year: the days the wind
    lets the system fly, taken as the wind-operability fraction times the
    calendar year, ``N_op = f_wind * 365``. It is availability-independent
    by design: the operator/maintenance crew is a standing cost tied to
    the operating calendar, whereas availability (downtime) reduces the
    flight hours and the delivered energy rather than the number of days
    the plant is manned.

    Args:
        performance (PerformanceData): System performance data.

    Returns:
        float: Annual operating days [days/year].
    """
    return DAYS_PER_YEAR * wind_operability_fraction(performance)


def annual_flight_hours(performance: PerformanceData,
                        availability: float = 1.0) -> float:
    """Annual operating (flight) hours [h/year].

    Args:
        performance (PerformanceData): System performance data.
        availability (float): Fraction of the operating-wind time the
            system is actually flown [-]. Defaults to 1.0.

    Returns:
        float: Annual flight hours [h/year].
    """
    fraction = np.trapezoid(performance.windPdf * _operating_mask(performance),
                            performance.windSpeeds)
    return HOURS_PER_YEAR * availability * float(fraction)


def annual_reelout_hours(performance: PerformanceData,
                         availability: float = 1.0,
                         load_weight: Optional[np.ndarray] = None
                         ) -> Optional[float]:
    """Annual reel-out (traction) hours [h/year].

    The reel-out hours are the flight hours weighted by the reel-out time
    fraction, integrated over the wind distribution. This is the loaded
    time a pumping canopy actually experiences.

    An optional per-wind-speed ``load_weight`` scales each bin's
    contribution, turning the plain hour integral into a
    damage-equivalent hour count (Miner's rule); with no weight every
    hour counts equally.

    Args:
        performance (PerformanceData): System performance data.
        availability (float): Fraction of the operating-wind time the
            system is actually flown [-]. Defaults to 1.0.
        load_weight (np.ndarray): Optional per-wind-speed weighting
            factor (e.g. ``(F/F_ref)^m``). Defaults to None (unweighted).

    Returns:
        float: Annual (damage-equivalent) reel-out hours [h/year], or
        None when the reel-out timing is unavailable.
    """
    if performance.reelOutTimeFraction is None:
        return None
    reelOutFraction = np.asarray(performance.reelOutTimeFraction, dtype=float)
    weight = (performance.windPdf * _operating_mask(performance) *
              reelOutFraction)
    if load_weight is not None:
        weight = weight * np.asarray(load_weight, dtype=float)
    return HOURS_PER_YEAR * availability * float(
        np.trapezoid(weight, performance.windSpeeds))


def annual_cycle_count(performance: PerformanceData,
                       availability: float = 1.0) -> float:
    """Annual number of pumping cycles [1/year].

    Estimated as the operating time divided by the pumping cycle duration,
    integrated over the wind distribution. Wind speeds where the cycle time
    is zero (system not operating) contribute no cycles.

    Args:
        performance (PerformanceData): System performance data.
        availability (float): Fraction of the operating-wind time the
            system is actually flown [-]. Defaults to 1.0.

    Returns:
        float: Annual pumping cycle count [1/year].
    """
    cycleTime = np.asarray(performance.cycleTime, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        cyclesPerHour = (_operating_mask(performance) *
                         SECONDS_PER_HOUR / cycleTime)
    cyclesPerHour = np.nan_to_num(cyclesPerHour, nan=0.0,
                                  posinf=0.0, neginf=0.0)
    count = np.trapezoid(performance.windPdf * cyclesPerHour,
                         performance.windSpeeds)
    return HOURS_PER_YEAR * availability * float(count)

"""Development-stage maturity presets for the cost model.

A single ``development_stage`` setting (``early``, ``mid`` or ``mature``)
selects a bundle of maturity-linked inputs, so an early-prototype or a
mature-system cost run can be produced without editing the individual
lifetimes and labour hours. Use ``none`` (or omit the setting) to run a
fully manual configuration where every lifetime, labour hour and tether
mode is taken from the config/cost files as written. Any numeric value
written explicitly in the config overrides the stage preset.

Selecting a stage models maturity through *operational wear*: by default it
puts the tether replacement into operational-life mode
(``operational_life_enabled`` on, bending ``master_curve`` off) so the staged
``operational_life_flight_hours`` governs. This isolates the reliability/maturity
effect from the tether bending *design* (safety factor). A cost file that sets
``operational_life_enabled: false`` opts out and keeps the bending master curve,
so the design study can run a stress-sensitive tether life at a chosen maturity
(not only at ``development_stage: none``).
"""

from typing import Dict, Optional

WEEKS_PER_YEAR = 52.0

VALID_STAGES = ('early', 'mid', 'mature')

# The base-crew operating effort is specified per week (a stable target that
# does not depend on the site), and converted to the per-day value the cost
# model needs using the wind resource's operating-day count N_op (see
# ``stage_operating_hours_per_day``). Everything else is a direct value.
STAGE_PRESETS: Dict[str, Dict[str, float]] = {
    'early': {
        'canopy_lifetime_flight_hours': 100.0,
        'operating_hours_per_week': 20.0,
        'maintenance_hours_per_flight_hour': 0.25,
        'operational_life_flight_hours': 250.0,
    },
    'mid': {
        'canopy_lifetime_flight_hours': 500.0,
        'operating_hours_per_week': 5.0,
        'maintenance_hours_per_flight_hour': 0.10,
        'operational_life_flight_hours': 1000.0,
    },
    'mature': {
        'canopy_lifetime_flight_hours': 5000.0,
        'operating_hours_per_week': 1.0,
        'maintenance_hours_per_flight_hour': 0.01,
        'operational_life_flight_hours': 5000.0,
    },
}


def stage_defaults(stage) -> Dict[str, float]:
    """Return the maturity input defaults for a development stage.

    Args:
        stage: Development stage name, or None/``'none'`` for a fully
            manual configuration. Defaults to an empty mapping in that
            case.

    Returns:
        A copy of the preset values for the stage, or an empty mapping
        when no stage is selected.

    Raises:
        ValueError: If the stage name is not one of ``VALID_STAGES``
            (or ``none``).
    """
    if stage is None:
        return {}
    if isinstance(stage, str) and stage.strip().lower() in ('', 'none'):
        return {}
    if stage not in STAGE_PRESETS:
        raise ValueError(
            f"development_stage must be one of {VALID_STAGES} or 'none', "
            f"got: {stage}")
    return dict(STAGE_PRESETS[stage])


def stage_operating_hours_per_day(
        stage_defaults: Dict[str, float],
        n_op: Optional[float]) -> Optional[float]:
    """Convert a stage's weekly operating hours to a per-day value.

    ``operating_hours_per_day = operating_hours_per_week * 52 / N_op``,
    where ``N_op`` is the wind resource's operating-day count per year, so
    the same weekly effort maps onto whatever operating-day count the site
    actually has.

    Args:
        stage_defaults: A stage preset mapping (or empty for no stage).
        n_op: Operating days per year of the wind resource.

    Returns:
        The per-day operating hours, or None when the mapping has no
        weekly-hours entry or ``n_op`` is missing/zero.
    """
    hours_per_week = (stage_defaults or {}).get('operating_hours_per_week')
    if hours_per_week is None or not n_op:
        return None
    return hours_per_week * WEEKS_PER_YEAR / n_op

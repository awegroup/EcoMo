"""Unit tests for AWESPA power curve parsing."""

import numpy as np
import pytest

from ecomo.ecomo_economic import _parse_power_curves


def _make_entry(windSpeed, successful, power, cycleTime):
    """Build one wind_speed_data entry in awesIO power curve format."""
    return {
        'wind_speed': windSpeed,
        'successful': successful,
        'performance': {
            'power': {'average_cycle_power': power},
            'timing': {'cycle_time': cycleTime},
        },
    }


def test_parse_power_curves_single_profile_direct_values():
    data = {
        'reference_wind_speeds': [5.0, 10.0, 15.0],
        'power_curves': [
            {
                'profile_id': 1,
                'probability_weight': 1.0,
                'wind_speed_data': [
                    _make_entry(5.0, False, 0.0, 0.0),
                    _make_entry(10.0, True, 1000.0, 100.0),
                    _make_entry(15.0, True, 3000.0, 90.0),
                ],
            },
        ],
    }
    parsed = _parse_power_curves(data)

    np.testing.assert_allclose(parsed['windRange'], [5.0, 10.0, 15.0])
    np.testing.assert_allclose(parsed['peAvg'], [0.0, 1000.0, 3000.0])
    np.testing.assert_allclose(parsed['dtCycle'], [0.0, 100.0, 90.0])
    assert parsed['peRated'] == 3000.0


def test_parse_power_curves_multi_profile_weighted_average():
    data = {
        'reference_wind_speeds': [5.0, 10.0],
        'power_curves': [
            {
                'profile_id': 1,
                'probability_weight': 0.75,
                'wind_speed_data': [
                    _make_entry(5.0, True, 400.0, 100.0),
                    _make_entry(10.0, True, 2000.0, 80.0),
                ],
            },
            {
                'profile_id': 2,
                'probability_weight': 0.25,
                'wind_speed_data': [
                    # Failed point counts as zero power
                    _make_entry(5.0, False, 0.0, 0.0),
                    _make_entry(10.0, True, 1000.0, 120.0),
                ],
            },
        ],
    }
    parsed = _parse_power_curves(data)

    # Power: failed point contributes zero, divided by total weight
    np.testing.assert_allclose(parsed['peAvg'],
                               [0.75 * 400.0, 0.75 * 2000.0 + 0.25 * 1000.0])
    # Cycle time: averaged over successful profiles only
    np.testing.assert_allclose(parsed['dtCycle'],
                               [100.0, 0.75 * 80.0 + 0.25 * 120.0])
    assert parsed['peRated'] == pytest.approx(1750.0)


def test_parse_power_curves_all_failed_wind_speed_gives_zero():
    data = {
        'reference_wind_speeds': [5.0],
        'power_curves': [
            {
                'profile_id': 1,
                'probability_weight': 0.5,
                'wind_speed_data': [_make_entry(5.0, False, 0.0, 0.0)],
            },
            {
                'profile_id': 2,
                'probability_weight': 0.5,
                'wind_speed_data': [_make_entry(5.0, False, 0.0, 0.0)],
            },
        ],
    }
    parsed = _parse_power_curves(data)

    assert parsed['peAvg'][0] == 0.0
    assert parsed['dtCycle'][0] == 0.0
    assert not np.isnan(parsed['dtCycle']).any()


def test_parse_power_curves_length_mismatch_raises():
    data = {
        'reference_wind_speeds': [5.0, 10.0],
        'power_curves': [
            {
                'profile_id': 1,
                'probability_weight': 1.0,
                'wind_speed_data': [_make_entry(5.0, True, 400.0, 100.0)],
            },
        ],
    }
    with pytest.raises(ValueError):
        _parse_power_curves(data)


def test_parse_power_curves_empty_raises():
    with pytest.raises(ValueError):
        _parse_power_curves({'reference_wind_speeds': [5.0],
                             'power_curves': []})

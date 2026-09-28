"""Unit tests for the YAML input loaders."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from ecomo.loaders.eco_load_cost_inputs import eco_load_cost_inputs
from ecomo.loaders.eco_load_system import eco_load_system
from ecomo.loaders.eco_load_performance import (
    eco_load_performance,
    peak_mechanical_power_fallback,
)
from ecomo.loaders.eco_load_wind_resource import eco_load_wind_resource
from ecomo.constants import (
    DrivetrainType,
    StorageType,
    WinchMaterial,
)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "test"
DATA_DIR = PROJECT_ROOT / "data"

# Each topology case lives in its own subfolder of config/test/
_TOPOLOGY_FOLDER = {"GG_fixed": "gg_fixed", "GG_soft": "gg_soft", "FG": "fg"}


def config_path(filename):
    """Resolve a config filename to its topology subfolder."""
    for case, folder in _TOPOLOGY_FOLDER.items():
        if filename.endswith(f"{case}.yml"):
            return CONFIG_DIR / folder / filename
    raise ValueError(f"Unknown config file: {filename}")


def test_cost_loader_translates_type_names_to_enums():
    costs = eco_load_cost_inputs(
        config_path("economic_cost_inputs_GG_fixed.yml"))

    assert costs.groundStation.winch.material == WinchMaterial.STEEL
    assert costs.groundStation.drivetrain == DrivetrainType.ELECTRIC
    assert costs.groundStation.electricalStorage == StorageType.ULTRACAPACITOR


def test_cost_loader_builds_typed_fields():
    costs = eco_load_cost_inputs(
        config_path("economic_cost_inputs_GG_fixed.yml"))

    assert costs.kite.priceStructuralMass == 250.0
    assert costs.tether.priceMass == 80
    assert isinstance(costs.tether.creepLifeCoefficients, np.ndarray)
    assert costs.balanceOfSystem.decommissioningInstallationFraction == 0.5
    assert costs.market.electricityPriceIntercept == 45.0
    assert costs.market.subsidy == 150.0


def test_cost_loader_applies_max_stress_override():
    costs = eco_load_cost_inputs(
        config_path("economic_cost_inputs_GG_fixed.yml"),
        tether_max_stress=2.0e9)

    assert costs.tether.maxStress == 2.0e9


def test_winch_material_properties_select_steel():
    costs = eco_load_cost_inputs(
        config_path("economic_cost_inputs_GG_fixed.yml"))
    winch = costs.groundStation.winch

    assert winch.price == winch.priceSteel
    assert winch.density == winch.densitySteel
    assert winch.maxStress == winch.maxStressSteel


def test_cost_loader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        eco_load_cost_inputs(tmp_path / "missing.yml")


# --------------------------------------------------------------------- #
# Bending fatigue coefficient a1 <-> winch D/d coupling
# --------------------------------------------------------------------- #

_A1_TABLE = {
    'drum_to_tether_ratio': [10, 20, 30, 100],
    'a1': [5.4, 5.8, 6.1, 6.5],
}


def _write_cost_with_a1_table(tmp_path, drum_ratio, table=_A1_TABLE):
    """Copy the GG-fixed cost config, swap the scalar a1 for the table."""
    with open(config_path("economic_cost_inputs_GG_fixed.yml"), 'r',
              encoding='utf-8') as f:
        data = yaml.safe_load(f)
    del data['costs']['tether']['bending_life_a1']
    if table is not None:
        data['costs']['tether']['bending_life_a1_table'] = table
    if drum_ratio is not None:
        data['costs']['ground_station']['winch'][
            'drum_to_tether_diameter_ratio'] = drum_ratio
    else:
        del data['costs']['ground_station']['winch'][
            'drum_to_tether_diameter_ratio']
    path = tmp_path / "cost.yml"
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)
    return path


def test_bending_a1_table_reproduces_calibration_points(tmp_path):
    # Every tabulated D/d must resolve to its exact a1 (interpolation
    # through, not around, the calibration data)
    for ratio, expected in zip(_A1_TABLE['drum_to_tether_ratio'],
                               _A1_TABLE['a1']):
        costs = eco_load_cost_inputs(
            _write_cost_with_a1_table(tmp_path, ratio))
        assert costs.tether.bendingLifeA1 == pytest.approx(expected)


def test_bending_a1_table_interpolates_between_points(tmp_path):
    costs = eco_load_cost_inputs(_write_cost_with_a1_table(tmp_path, 50))
    # Strictly between the D/d=30 (a1=6.1) and D/d=100 (a1=6.5) anchors
    assert 6.1 < costs.tether.bendingLifeA1 < 6.5


def test_bending_a1_table_smaller_drum_gives_shorter_life(tmp_path):
    # Monotonic: a smaller D/d (tighter bend) must not increase a1
    small = tmp_path / "small"
    large = tmp_path / "large"
    small.mkdir()
    large.mkdir()
    costsSmall = eco_load_cost_inputs(_write_cost_with_a1_table(small, 20))
    costsLarge = eco_load_cost_inputs(_write_cost_with_a1_table(large, 100))
    assert costsSmall.tether.bendingLifeA1 < costsLarge.tether.bendingLifeA1


def test_bending_a1_table_warns_when_extrapolating(tmp_path):
    with pytest.warns(UserWarning, match="extrapolated"):
        eco_load_cost_inputs(_write_cost_with_a1_table(tmp_path, 200))


def test_bending_a1_table_requires_drum_ratio(tmp_path):
    with pytest.raises(KeyError, match="drum_to_tether_diameter_ratio"):
        eco_load_cost_inputs(_write_cost_with_a1_table(tmp_path, None))


def _write_cost_with_operational(tmp_path, enabled, hours=250.0):
    """GG-fixed cost config with an operational-life toggle set."""
    with open(config_path("economic_cost_inputs_GG_fixed.yml"), 'r',
              encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if enabled is not None:
        data['costs']['tether']['operational_life_enabled'] = enabled
    data['costs']['tether']['operational_life_flight_hours'] = hours
    path = tmp_path / "cost.yml"
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)
    return path


def test_operational_life_disabled_by_default(tmp_path):
    # No toggle key -> operational mode off (life = None)
    costs = eco_load_cost_inputs(_write_cost_with_operational(tmp_path, None))
    assert costs.tether.operationalLife is None


def test_operational_life_toggle_off_ignores_value(tmp_path):
    costs = eco_load_cost_inputs(
        _write_cost_with_operational(tmp_path, False, hours=250.0))
    assert costs.tether.operationalLife is None


def test_operational_life_toggle_on_uses_value(tmp_path):
    costs = eco_load_cost_inputs(
        _write_cost_with_operational(tmp_path, True, hours=250.0))
    assert costs.tether.operationalLife == pytest.approx(250.0)


def test_operational_life_enabled_requires_value(tmp_path):
    with open(config_path("economic_cost_inputs_GG_fixed.yml"), 'r',
              encoding='utf-8') as f:
        data = yaml.safe_load(f)
    data['costs']['tether']['operational_life_enabled'] = True
    data['costs']['tether'].pop('operational_life_flight_hours', None)
    path = tmp_path / "cost.yml"
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)
    with pytest.raises(KeyError, match="operational_life_flight_hours"):
        eco_load_cost_inputs(path)


def test_bending_a1_explicit_scalar_overrides_table(tmp_path):
    # An explicit scalar always wins, even with a table also present
    with open(config_path("economic_cost_inputs_GG_fixed.yml"), 'r',
              encoding='utf-8') as f:
        data = yaml.safe_load(f)
    data['costs']['tether']['bending_life_a1_table'] = _A1_TABLE
    data['costs']['ground_station']['winch'][
        'drum_to_tether_diameter_ratio'] = 20  # would give 5.8 from the table
    path = tmp_path / "cost.yml"
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)

    costs = eco_load_cost_inputs(path)
    assert costs.tether.bendingLifeA1 == pytest.approx(6.5)  # the scalar


def test_system_loader_reads_physical_parameters():
    systemData = eco_load_system(config_path("system_GG_fixed.yml"),
                                 wing="fixed")

    assert systemData['kiteMass'] == 5543
    assert systemData['kiteFlatArea'] == 100
    assert systemData['tether']['d'] == pytest.approx(0.0273)
    assert systemData['tether']['L'] == pytest.approx(2600.0)
    assert systemData['tetherMaxStress'] == pytest.approx(1.5e9)
    # Storage capacities are converted from Wh (awesIO) to kWh
    assert systemData['storageCapacities']['ultracapacitor'] == (
        pytest.approx(11.25))
    assert systemData['storageCapacities']['battery'] == (
        pytest.approx(1000.0))


def test_system_loader_resolves_span_aspect_ratio_area():
    # The FG example wing provides span and aspect ratio but no flat
    # wing area; the flat area is derived as span^2 / aspect_ratio
    systemData = eco_load_system(config_path("system_FG.yml"),
                                 wing="fixed")

    assert systemData['kiteSpan'] == pytest.approx(10.0)
    assert systemData['kiteFlatArea'] == pytest.approx(10.0 ** 2 / 6.0)


def test_system_loader_soft_kite_flat_area_fallback(tmp_path):
    # Without flat_wing_area, span or aspect ratio, the flat area of a
    # soft kite is estimated from the projected area with the 25/18
    # factor
    with open(config_path("system_GG_soft.yml"), 'r',
              encoding='utf-8') as f:
        data = yaml.safe_load(f)
    wingStructure = data['components']['kites'][0]['wing']['structure']
    del wingStructure['flat_wing_area']
    systemPath = tmp_path / "system.yml"
    with open(systemPath, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)

    with pytest.warns(UserWarning, match="flat_wing_area"):
        systemData = eco_load_system(systemPath, wing="soft")

    projectedArea = wingStructure['projected_surface_area']
    assert systemData['kiteFlatArea'] == pytest.approx(
        25 / 18 * projectedArea)


def test_performance_loader_reads_arrays():
    perf = eco_load_performance(
        config_path("system_performance_GG_fixed.yml"), 'GG')

    assert len(perf['windRange']) == 25
    assert len(perf['peAvg']) == 25
    assert perf['peRated'] == pytest.approx(1.0e6)
    assert perf['pmPeak'] == pytest.approx(1.87e6)
    assert len(perf['tetherForce']) == 25
    assert 'ultracapacitor' in perf['exchangedEnergy']


def test_wind_resource_loader_returns_normalized_density():
    binCenters, density = eco_load_wind_resource(
        DATA_DIR / "power_law_case_1.yml")

    assert len(binCenters) == len(density)
    assert np.all(density >= 0)
    assert np.all(np.isfinite(density))
    # The marginal distribution is a probability density: it integrates
    # to approximately one over the wind-speed bins
    assert np.trapezoid(density, binCenters) == pytest.approx(1.0, abs=0.05)


def test_wind_resource_loader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        eco_load_wind_resource(tmp_path / "missing.yml")


def test_performance_loader_length_mismatch_raises(tmp_path):
    performancePath = tmp_path / "performance.yml"
    with open(performancePath, 'w', encoding='utf-8') as f:
        yaml.safe_dump({
            'reference_wind_speeds': [5.0, 10.0],
            'rated_electrical_power': 1000.0,
            'average_cycle_power': [100.0, 500.0, 900.0],
        }, f)

    with pytest.raises(ValueError, match="average_cycle_power"):
        eco_load_performance(performancePath, 'FG')


# --------------------------------------------------------------------- #
#  Warnings for absent optional fields
# --------------------------------------------------------------------- #

def _write_performance(tmp_path, **overrides):
    """Write a minimal performance YAML, with field overrides/removals.

    A field set to ``None`` in ``overrides`` is omitted from the file.
    """
    data = {
        'reference_wind_speeds': [5.0, 10.0],
        'rated_electrical_power': 1.0e6,
        'peak_mechanical_power': 1.5e6,
        'average_cycle_power': [1.0e5, 5.0e5],
        'cycle_time': [100.0, 100.0],
        'tether_force': [1.0e4, 2.0e4],
        'tip_speed_ratio': 7.0,
    }
    data.update(overrides)
    data = {k: v for k, v in data.items() if v is not None}
    path = tmp_path / "performance.yml"
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)
    return path


def test_performance_warns_when_tether_force_absent(tmp_path):
    path = _write_performance(tmp_path, tether_force=None)
    with pytest.warns(UserWarning, match="tether_force"):
        perf = eco_load_performance(path, 'GG')
    assert perf['tetherForce'] is None


def test_performance_warns_and_falls_back_for_gg_cycle_time(tmp_path):
    path = _write_performance(tmp_path, cycle_time=None)
    with pytest.warns(UserWarning, match="cycle_time"):
        perf = eco_load_performance(path, 'GG')
    assert perf['dtCycle'] == pytest.approx(60.0)


def test_performance_no_cycle_time_warning_for_fg(tmp_path):
    # FG systems do not use the cycle time, so its absence is expected
    import warnings as _warnings
    path = _write_performance(tmp_path, cycle_time=None, tether_force=[1.0, 2.0],
                              tip_speed_ratio=7.0)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        perf = eco_load_performance(path, 'FG')
    assert perf['dtCycle'] is None


def test_performance_warns_when_fg_tip_speed_ratio_absent(tmp_path):
    path = _write_performance(tmp_path, tip_speed_ratio=None)
    with pytest.warns(UserWarning, match="tip_speed_ratio"):
        eco_load_performance(path, 'FG')


def test_performance_peak_mechanical_power_fallback_warns():
    with pytest.warns(UserWarning, match="peak_mechanical_power"):
        result = peak_mechanical_power_fallback(1.0e6)
    assert result == pytest.approx(2.5e6)


def test_system_warns_when_breaking_strength_absent(tmp_path):
    with open(config_path("system_GG_fixed.yml"), 'r',
              encoding='utf-8') as f:
        data = yaml.safe_load(f)
    del data['components']['tethers'][0]['structure']['material'][
        'breaking_strength']
    path = tmp_path / "system.yml"
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)

    with pytest.warns(UserWarning, match="breaking_strength"):
        systemData = eco_load_system(path, wing="fixed")
    assert systemData['tetherMaxStress'] is None


def test_cost_warns_when_onboard_battery_absent_for_gg(tmp_path):
    with open(config_path("economic_cost_inputs_GG_fixed.yml"), 'r',
              encoding='utf-8') as f:
        data = yaml.safe_load(f)
    del data['costs']['kite']['onboard_battery']
    path = tmp_path / "cost.yml"
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)

    with pytest.warns(UserWarning, match="onboard_battery"):
        eco_load_cost_inputs(path, power='GG')


def test_cost_no_onboard_battery_warning_without_power(tmp_path):
    # Without a topology, the loader cannot know the field is needed
    import warnings as _warnings
    with open(config_path("economic_cost_inputs_GG_fixed.yml"), 'r',
              encoding='utf-8') as f:
        data = yaml.safe_load(f)
    del data['costs']['kite']['onboard_battery']
    path = tmp_path / "cost.yml"
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        eco_load_cost_inputs(path)


def test_cost_warns_when_hydraulic_components_absent(tmp_path):
    with open(config_path("economic_cost_inputs_GG_fixed.yml"), 'r',
              encoding='utf-8') as f:
        data = yaml.safe_load(f)
    gs = data['costs']['ground_station']
    gs['drivetrain'] = 'hydraulic'
    for key in ('pump_motor', 'hydraulic_accumulator', 'hydraulic_motor'):
        gs.pop(key, None)
    path = tmp_path / "cost.yml"
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f)

    with pytest.warns(UserWarning, match="pump_motor"):
        eco_load_cost_inputs(path, power='GG')

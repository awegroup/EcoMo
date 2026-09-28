"""Tests for the development-stage maturity presets and their wiring."""

from pathlib import Path

import pytest
import yaml

from ecomo import EcoMo
from ecomo.maturity import STAGE_PRESETS, VALID_STAGES, stage_defaults
from ecomo.loaders import eco_load_cost_inputs
from ecomo.loaders.eco_load_cost_inputs import _apply_stage_defaults_to_costs

PROJECT_ROOT = Path(__file__).parent.parent
COST_INPUTS = (PROJECT_ROOT / "config" / "test" / "gg_soft" /
               "economic_cost_inputs_GG_soft.yml")


def test_stage_defaults_returns_expected_mid_preset():
    defaults = stage_defaults("mid")
    assert defaults["canopy_lifetime_flight_hours"] == 500.0
    assert defaults["operating_hours_per_week"] == 5.0
    assert defaults["maintenance_hours_per_flight_hour"] == 0.10
    assert defaults["operational_life_flight_hours"] == 1000.0


def test_stage_defaults_covers_all_valid_stages():
    for stage in VALID_STAGES:
        assert set(stage_defaults(stage)) == set(STAGE_PRESETS[stage])


def test_stage_defaults_none_is_empty():
    assert stage_defaults(None) == {}


def test_stage_defaults_none_string_is_empty():
    # 'none' (any case) selects a fully manual configuration, like null.
    assert stage_defaults("none") == {}
    assert stage_defaults("None") == {}


def test_stage_defaults_invalid_stage_raises():
    with pytest.raises(ValueError, match="development_stage"):
        stage_defaults("prototype")


def test_stage_defaults_are_copies_not_shared_state():
    defaults = stage_defaults("early")
    defaults["canopy_lifetime_flight_hours"] = 1.0
    assert STAGE_PRESETS["early"]["canopy_lifetime_flight_hours"] == 100.0


def _cost_yaml(tmp_path, canopy=None, enable_operational=False):
    """Write a soft-wing cost YAML copy with selected maturity fields.

    Args:
        tmp_path: pytest temporary directory.
        canopy: Explicit canopy flight-hour life to set, or None to leave
            it absent (calendar model).
        enable_operational: Whether to enable the operational tether life
            (its flight-hour value is always removed so the stage can fill
            it).
    """
    data = yaml.safe_load(COST_INPUTS.read_text(encoding="utf-8"))
    soft = data["costs"]["kite"]["structure"]["soft"]
    soft.pop("canopy_lifetime_flight_hours", None)
    if canopy is not None:
        soft["canopy_lifetime_flight_hours"] = canopy
    tether = data["costs"]["tether"]
    tether["operational_life_enabled"] = enable_operational
    tether.pop("operational_life_flight_hours", None)
    path = tmp_path / "cost_inputs.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_stage_fills_canopy_life_when_absent(tmp_path):
    path = _cost_yaml(tmp_path, canopy=None)
    costs = eco_load_cost_inputs(path, power="GG",
                                 stage_defaults=stage_defaults("mid"))
    assert costs.kite.canopyLifetimeFlightHours == 500.0


def test_explicit_canopy_life_overrides_stage(tmp_path):
    # The file keeps its own canopy life -> the stage must not override it.
    path = _cost_yaml(tmp_path, canopy=3333.0)
    costs = eco_load_cost_inputs(path, power="GG",
                                 stage_defaults=stage_defaults("mature"))
    assert costs.kite.canopyLifetimeFlightHours == 3333.0


def test_stage_fills_tether_operational_life_when_enabled(tmp_path):
    path = _cost_yaml(tmp_path, enable_operational=True)
    costs = eco_load_cost_inputs(path, power="GG",
                                 stage_defaults=stage_defaults("mid"))
    assert costs.tether.operationalLife == 1000.0


def test_no_stage_leaves_canopy_absent(tmp_path):
    path = _cost_yaml(tmp_path, canopy=None)
    # Without a stage the canopy life is genuinely absent -> the calendar
    # lifetime model is used instead (no flight-hour life set).
    costs = eco_load_cost_inputs(path, power="GG", stage_defaults=None)
    assert costs.kite.canopyLifetimeFlightHours is None


def _tether_data(master_curve=True, operational_enabled=False):
    """Minimal cost dict with a tether block for stage-mode tests."""
    tether = {"operational_life_enabled": operational_enabled}
    if master_curve:
        tether["master_curve"] = {"coefficient": 5.0e6, "exponent": 2.0}
    return {"costs": {"tether": tether}}


def test_explicit_operational_off_opts_out_of_stage_forcing():
    # A file that explicitly sets operational_life_enabled: false opts out of
    # the stage's operational-wear forcing: the bending master curve is kept
    # (so tether life stays stress-sensitive for the design study), while the
    # staged operational_life_flight_hours is still filled for reference.
    data = _tether_data(master_curve=True, operational_enabled=False)
    _apply_stage_defaults_to_costs(data, stage_defaults("mid"))
    tether = data["costs"]["tether"]
    assert tether["master_curve"] == {"coefficient": 5.0e6, "exponent": 2.0}
    assert tether["operational_life_enabled"] is False
    assert tether["operational_life_flight_hours"] == 1000.0


def test_stage_forces_operational_tether_mode_by_default():
    # A file that does NOT opt out (operational life on, as the maturity-roadmap
    # cost file has it): selecting a stage keeps operational-wear mode -- master
    # curve off, operational life enabled and filled from the preset.
    data = _tether_data(master_curve=True, operational_enabled=True)
    _apply_stage_defaults_to_costs(data, stage_defaults("mid"))
    tether = data["costs"]["tether"]
    assert tether["master_curve"] is None
    assert tether["operational_life_enabled"] is True
    assert tether["operational_life_flight_hours"] == 1000.0


def test_no_stage_leaves_tether_mode_untouched():
    # With stage 'none' the file's bending master curve is preserved.
    data = _tether_data(master_curve=True, operational_enabled=False)
    _apply_stage_defaults_to_costs(data, stage_defaults("none"))
    tether = data["costs"]["tether"]
    assert tether["master_curve"] == {"coefficient": 5.0e6, "exponent": 2.0}
    assert tether["operational_life_enabled"] is False
    assert "operational_life_flight_hours" not in tether


# A round operating-day count so the mid weekly target (5 h/wk) converts to
# a clean 1.0 h/day: 5 * 52 / 260 = 1.0.
N_OP = 260.0


def _operations_model(stage, **operations):
    model = EcoMo()
    model.settings = {"operations": {"labour_price": 50, "automation": 0.0,
                                     **operations}}
    model.stageDefaults = stage_defaults(stage)
    return model


def test_operations_use_stage_hours_when_absent():
    # 5 h/wk (mid) * 52 / 260 d/yr = 1.0 h/day.
    ops = _operations_model("mid")._build_operations(N_OP)
    assert ops.operatingHoursPerDay == 1.0
    assert ops.maintenanceHoursPerFlightHour == 0.10


def test_operations_stage_hours_scale_with_n_op():
    # Halving the operating-day count doubles the per-day hours.
    ops = _operations_model("mid")._build_operations(N_OP / 2)
    assert ops.operatingHoursPerDay == 2.0


def test_operations_explicit_hours_override_stage():
    ops = _operations_model(
        "mid", operating_hours_per_day=4.0)._build_operations(N_OP)
    assert ops.operatingHoursPerDay == 4.0            # explicit wins
    assert ops.maintenanceHoursPerFlightHour == 0.10  # from the stage


def test_operations_without_hours_or_stage_raises():
    with pytest.raises(ValueError, match="development_stage"):
        _operations_model(None)._build_operations(N_OP)

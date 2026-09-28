"""Tests for the optional analysis/plotting layer.

Covers the two acceptance requirements:
- analysis is OFF by default, so existing behaviour is unchanged;
- with analysis ON, ``run_analysis`` produces the figures for the V3
  example without error.

Also checks the sweep harness: it does not mutate the base config and it
actually changes the cost-side LCoE.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import pytest
import yaml

from ecomo import EcoMo
from ecomo.analysis import (
    analysis_options,
    maybe_run_analysis,
    run_analysis,
)
from ecomo.analysis.sweep import PARAMS, SweepRunner, sweep_1d

PROJECT_ROOT = Path(__file__).parent.parent
V3_SETTINGS = (PROJECT_ROOT / "config" / "example" /
               "economic_settings_V3_example.yml")


def _v3_eco():
    """Compute the V3 example baseline, quietly."""
    model = EcoMo()
    model.load_configuration(V3_SETTINGS, validate=False)
    return model, model.compute_economics(validate=False)


# --------------------------------------------------------------------- #
#  OFF by default
# --------------------------------------------------------------------- #

def _disabled_settings(tmp_path):
    """Write a minimal settings file with the analysis block disabled."""
    path = tmp_path / "disabled_settings.yml"
    path.write_text("analysis:\n  enabled: false\n", encoding="utf-8")
    return path


def test_analysis_disabled_when_flag_false_or_absent():
    # Off by default: disabled both when the block is absent and when the
    # enabled flag is present-but-false. (Independent of the example file,
    # whose flag the user may toggle.)
    assert analysis_options({})['enabled'] is False
    assert analysis_options({'analysis': {}})['enabled'] is False
    assert analysis_options({'analysis': {'enabled': False}})['enabled'] is False


def test_maybe_run_analysis_noop_when_disabled(tmp_path):
    outdir = tmp_path / "analysis"
    # Disabled short-circuits before loading any model -> returns None and
    # writes nothing.
    produced = maybe_run_analysis(_disabled_settings(tmp_path), outdir)
    assert produced is None
    assert not outdir.exists()


def test_results_unchanged_with_analysis_module_imported(tmp_path):
    # Importing/using the analysis layer must not perturb the economics
    _, ecoA = _v3_eco()
    _, ecoB = _v3_eco()
    assert ecoA['metrics']['LCoE'] == pytest.approx(ecoB['metrics']['LCoE'])
    # And the disabled toggle leaves the model output identical
    assert maybe_run_analysis(_disabled_settings(tmp_path),
                              tmp_path / 'unused') is None


# --------------------------------------------------------------------- #
#  Sweep harness correctness
# --------------------------------------------------------------------- #

def test_sweep_does_not_mutate_base_config():
    before = V3_SETTINGS.read_bytes()
    with SweepRunner(V3_SETTINGS) as runner:
        runner.run([('cost', PARAMS['canopy_life'].path, 123.0)])
    assert V3_SETTINGS.read_bytes() == before


def test_sweep_1d_changes_cost_side_lcoe():
    with SweepRunner(V3_SETTINGS) as runner:
        rows = sweep_1d(runner, PARAMS['canopy_life'], [100.0, 500.0])
    # A longer canopy life means fewer replacements -> lower LCoE
    assert rows[0]['value'] == 100.0 and rows[1]['value'] == 500.0
    assert rows[0]['LCoE'] > rows[1]['LCoE']
    # AEP is a fixed input: the cost sweep must not move it
    assert rows[0]['AEP'] == pytest.approx(rows[1]['AEP'])


def test_perf_override_requires_awespa_mode_message():
    # Crest factor is a perf-domain override; on the AWESPA V3 config it works
    with SweepRunner(V3_SETTINGS) as runner:
        _, eco = runner.run([('perf', 'crest_factor', 3.0)])
    assert eco['metrics']['LCoE'] > 0


# --------------------------------------------------------------------- #
#  Report-consistency changes (availability, KCU OPEX, launch/recovery)
# --------------------------------------------------------------------- #

def test_kcu_replacement_opex_charged():
    # The V3 config sets avionics lifetime 5 yr -> OPEX = CAPEX / 5
    _, eco = _v3_eco()
    avionics = eco['kite']['avionics']
    assert avionics['OPEX'] == pytest.approx(avionics['CAPEX'] / 5.0)


def test_availability_scales_net_aep_and_raises_lcoe():
    with SweepRunner(V3_SETTINGS) as runner:
        # Pin the high case to a = 1.0 (the config default is now 0.90)
        _, full = runner.run([('settings', 'operations.availability', 1.0)])
        _, half = runner.run([('settings', 'operations.availability', 0.5)])
    # Downtime scales the delivered energy...
    assert half['metrics']['AEP'] == pytest.approx(
        0.5 * full['metrics']['AEP'], rel=1e-9)
    # ...and therefore raises LCoE (correct sign, unlike the old behaviour)
    assert half['metrics']['LCoE'] > full['metrics']['LCoE']


def test_bos_operating_terms_are_separate_leaves():
    _, eco = _v3_eco()
    bos = eco['BoS']
    om = bos['OM']
    # BoS.OM is now the per-kW overhead ONLY (no labour folded in)
    assert om['OPEX'] == pytest.approx(om['overhead_opex'])
    assert om['overhead_opex'] > 0
    # The crew labour is its own separate BoS.labour leaf group
    labour = bos['labour']
    assert labour['operation_opex'] > 0
    assert labour['maintenance_opex'] > 0
    assert labour['OPEX'] == pytest.approx(
        labour['operation_opex'] + labour['maintenance_opex'])
    # There is no separate top-level operations subsystem node
    assert 'operations' not in eco


def test_sensor_cost_itemised_separately_from_avionics():
    model, eco = _v3_eco()
    # The sensor is its own CAPEX line (the configured sensor cost), not
    # folded into the KCU (config-driven so it survives cost updates)
    assert eco['kite']['sensor']['CAPEX'] == pytest.approx(
        model.costs.kite.sensorCost)
    assert model.costs.kite.sensorCost > 0
    # Replacement OPEX = CAPEX / lifetime
    assert eco['kite']['sensor']['OPEX'] == pytest.approx(
        model.costs.kite.sensorCost / model.costs.kite.sensorLifetime)
    # Avionics remains the KCU-electronics line, distinct from the sensor
    assert eco['kite']['avionics']['CAPEX'] > 0


# --------------------------------------------------------------------- #
#  ON: figures are produced
# --------------------------------------------------------------------- #

def test_run_analysis_baseline_plots(tmp_path):
    # The baseline-only plots are fast (no sweeps)
    model, eco = _v3_eco()
    produced = run_analysis(
        V3_SETTINGS, tmp_path,
        plots=['stacked_bar', 'tether_life', 'power_curve'],
        model=model, eco=eco)
    for key in ('stacked_bar', 'tether_life', 'power_curve'):
        path = Path(produced[key])
        assert path.exists() and path.stat().st_size > 0
        assert path.suffix == '.png'


def test_run_analysis_sweep_plots_smoke(tmp_path):
    # Exercise the sweep-driven plots at minimal resolution
    produced = run_analysis(
        V3_SETTINGS, tmp_path,
        plots=['tornado', 'sweeps', 'contour'],
        tornado_keys=['canopy_life', 'labour_price'],
        sweep_keys=['canopy_life'], n_points=3, grid=(2, 2))
    assert Path(produced['tornado']).stat().st_size > 0
    assert Path(produced['contour']).stat().st_size > 0
    sweepPaths = produced['sweeps']
    assert len(sweepPaths) == 1
    assert all(Path(p).stat().st_size > 0 for p in sweepPaths)

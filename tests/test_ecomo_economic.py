"""Integration and regression tests for the EcoMo class."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from ecomo import EcoMo

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config" / "test"
DATA_DIR = PROJECT_ROOT / "data"

# Each topology case lives in its own subfolder of config/test/
_TOPOLOGY_FOLDER = {"GG_fixed": "gg_fixed", "GG_soft": "gg_soft", "FG": "fg"}
GG_FIXED_DIR = CONFIG_DIR / "gg_fixed"


def _settings_path(settingsName):
    """Resolve a settings filename to its topology subfolder."""
    for case, folder in _TOPOLOGY_FOLDER.items():
        if settingsName.endswith(f"{case}.yml"):
            return CONFIG_DIR / folder / settingsName
    raise ValueError(f"Unknown settings file: {settingsName}")

# Regression baseline for the GG fixed-wing example case.
# Updated for the foundation correction (Change 4): the foundation is now
# sized by the peak mechanical power (1870 kW) instead of the rated power
# (1000 kW), raising ICC by exactly 55 EUR/kW x 870 kW = 47850 EUR.
# Updated again after removing the ad-hoc BEND_LIFE_CORRECTION (=3) from the
# bending life model: the replacement frequency now follows Joshi & Trevisi
# (2024) Eq. 16 exactly, so it deliberately deviates from the MATLAB AWE-Eco
# reference (which divides by 3). Bending governs this case, so its f_repl
# triples (1.53 -> 4.59 /yr), raising OMC and LCoE; the IRR no longer
# converges (annual net revenue is negative).
BASELINE_GG_FIXED = {
    'LCoE': 277.839214,
    'NPV': -3528050.7211,
    'ICC': 3483013.6151,
    'OMC': 623556.2366,
    'AEP': 3398.672927,
    'CF': 0.387976,
    'IRR': float('nan'),
}


def _load_model(settingsName):
    model = EcoMo()
    model.load_configuration(
        economic_settings_path=_settings_path(settingsName),
    )
    return model


def _write_settings_copy(tmp_path, modify):
    """Copy the GG fixed settings to tmp_path with absolute file refs.

    Args:
        tmp_path (Path): Temporary directory for the copy.
        modify (callable): Function applied to the settings dict
            before writing.

    Returns:
        Path: Path to the written settings file.
    """
    with open(GG_FIXED_DIR / "economic_settings_GG_fixed.yml", 'r',
              encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    # File references resolve relative to the settings file; make the
    # non-null ones absolute so the copy works from tmp_path
    settings['input_files'] = {
        key: (str((GG_FIXED_DIR / value).resolve()) if value else value)
        for key, value in settings['input_files'].items()
    }
    modify(settings)
    settingsPath = tmp_path / "settings.yml"
    with open(settingsPath, 'w', encoding='utf-8') as f:
        yaml.safe_dump(settings, f)
    return settingsPath


def test_standalone_gg_fixed_matches_baseline():
    model = _load_model("economic_settings_GG_fixed.yml")
    eco = model.compute_economics()

    for key, ref in BASELINE_GG_FIXED.items():
        assert float(eco['metrics'][key]) == pytest.approx(
            ref, rel=1e-5, nan_ok=True), key


@pytest.mark.parametrize("settingsName", [
    "economic_settings_GG_fixed.yml",
    "economic_settings_GG_soft.yml",
    "economic_settings_FG.yml",
])
def test_standalone_all_topologies_produce_finite_metrics(settingsName):
    model = _load_model(settingsName)
    eco = model.compute_economics()

    metrics = eco['metrics']
    for key in ('LCoE', 'NPV', 'ICC', 'OMC', 'AEP', 'CF'):
        assert np.isfinite(metrics[key]), key
    assert metrics['LCoE'] > 0
    assert metrics['ICC'] > 0


def test_standalone_writes_results_yaml(tmp_path):
    model = _load_model("economic_settings_GG_fixed.yml")
    outputPath = tmp_path / "ecomo_results.yml"
    eco = model.compute_economics(output_path=outputPath)

    assert outputPath.exists()
    with open(outputPath, 'r', encoding='utf-8') as f:
        results = yaml.safe_load(f)

    assert results['metadata']['awesIO_version'] == '0.1.0'
    assert results['topology'] == {'power': 'GG', 'wing': 'fixed'}
    assert results['metrics']['lcoe_eur_per_mwh'] == pytest.approx(
        float(eco['metrics']['LCoE']))
    assert 'capex_eur' in results['cost_breakdown']


def _set_awespa_paths(settings):
    """Point the settings at the AWESPA output files (absolute paths)."""
    settings['input_files']['aep_results'] = str(
        (DATA_DIR / "aep_results.yml").resolve())
    settings['input_files']['power_curves'] = str(
        (DATA_DIR / "power_curves.yml").resolve())


def test_awespa_connected_mode_runs_without_nan(tmp_path):
    settingsPath = _write_settings_copy(tmp_path, _set_awespa_paths)

    # AWESPA mode is selected from the settings file. The tether force
    # is taken from the power curves, so no override is needed and no
    # "tether force" warning is raised.
    model = EcoMo()
    model.load_configuration(economic_settings_path=settingsPath)
    eco = model.compute_economics(output_path=tmp_path / "results.yml")

    metrics = eco['metrics']
    for key in ('LCoE', 'NPV', 'ICC', 'OMC', 'AEP', 'CF'):
        assert np.isfinite(metrics[key]), key
    assert metrics['LCoE'] > 0

    # The tether force comes from the power curves (a non-constant array)
    fT = model.inputs.performance.tetherForce
    assert np.any(fT > 0)
    assert np.ptp(fT) > 0

    # AEP must be taken directly from the AWESPA aep_results.yml file
    with open(DATA_DIR / "aep_results.yml", 'r', encoding='utf-8') as f:
        aepData = yaml.safe_load(f)
    assert metrics['AEP'] == pytest.approx(
        aepData['annual_energy_production']['total']['aep_mwh'])
    # ECOMo derives CF as AEP / (rated power x 8760), which agrees with
    # the AWESPA-reported capacity factor to within its own (slightly
    # different) internal definition -- a loose tolerance because the two
    # definitions of "rated power" differ marginally.
    assert metrics['CF'] == pytest.approx(
        aepData['power_summary']['capacity_factor'], rel=2e-2)


def test_awespa_mode_uses_scalar_tether_force_override(tmp_path):
    def modify(settings):
        settings['system_extras']['tether_force_override'] = 5000.0
        _set_awespa_paths(settings)

    settingsPath = _write_settings_copy(tmp_path, modify)

    model = EcoMo()
    model.load_configuration(economic_settings_path=settingsPath)
    eco = model.compute_economics()

    fT = model.inputs.performance.tetherForce
    assert len(fT) == len(model.inputs.performance.windSpeeds)
    np.testing.assert_allclose(fT, 5000.0)
    assert np.isfinite(eco['metrics']['LCoE'])


def test_load_configuration_rejects_unknown_topology(tmp_path):
    settingsPath = tmp_path / "bad_settings.yml"
    with open(settingsPath, 'w', encoding='utf-8') as f:
        yaml.safe_dump({
            'metadata': {'name': 'bad'},
            'topology': {'power': 'XX', 'wing': 'fixed'},
            'input_files': {'system': 'x.yml', 'cost_inputs': 'y.yml'},
        }, f)

    model = EcoMo()
    with pytest.raises(ValueError, match="topology.power"):
        model.load_configuration(economic_settings_path=settingsPath)


def test_wind_resource_file_changes_standalone_result(tmp_path):
    # The site wind distribution must drive the economic integrals; the
    # Weibull fallback and the site file give different metrics
    weibullModel = _load_model("economic_settings_GG_fixed.yml")
    weibullEco = weibullModel.compute_economics()
    assert weibullModel.windResource is None

    def set_resource(settings):
        settings['wind_resource']['resource_file'] = str(
            (DATA_DIR / "power_law_case_1.yml").resolve())

    settingsPath = _write_settings_copy(tmp_path, set_resource)
    siteModel = EcoMo()
    siteModel.load_configuration(economic_settings_path=settingsPath)
    siteEco = siteModel.compute_economics()

    assert siteModel.windResource is not None
    assert np.isfinite(siteEco['metrics']['LCoE'])
    assert siteEco['metrics']['LCoE'] != pytest.approx(
        weibullEco['metrics']['LCoE'])


def test_load_configuration_rejects_partial_awespa_files(tmp_path):
    def only_aep(settings):
        settings['input_files']['aep_results'] = str(
            (DATA_DIR / "aep_results.yml").resolve())

    settingsPath = _write_settings_copy(tmp_path, only_aep)

    model = EcoMo()
    with pytest.raises(ValueError, match="must both be set or both be null"):
        model.load_configuration(economic_settings_path=settingsPath)


def test_load_configuration_requires_input_files(tmp_path):
    settingsPath = tmp_path / "bad_settings.yml"
    with open(settingsPath, 'w', encoding='utf-8') as f:
        yaml.safe_dump({
            'metadata': {'name': 'bad'},
            'topology': {'power': 'GG', 'wing': 'fixed'},
        }, f)

    model = EcoMo()
    with pytest.raises(ValueError, match="input_files"):
        model.load_configuration(economic_settings_path=settingsPath)


def test_compute_economics_requires_configuration():
    model = EcoMo()
    with pytest.raises(ValueError, match="load_configuration"):
        model.compute_economics()

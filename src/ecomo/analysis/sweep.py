"""Reusable sweep harness for the EcoMo analysis layer.

Runs :class:`~ecomo.ecomo_economic.EcoMo` repeatedly while varying one or
two input parameters, collecting a tidy table of the resulting metrics and
per-subsystem cost breakdown. Each evaluation deep-copies the base
configuration into a private temporary directory, so the base config on
disk is never mutated and sweeps cannot leak state into one another.

Caveat: EcoMo consumes the performance (AEP, tether force, power curve) as
a fixed input, so a sweep changes only the cost-side response. Sweeps over
pure cost/assumption parameters are fully valid; sweeps over design
variables that also change the performance (wing area, generator sizing,
tether diameter) are valid only as "cost at fixed performance" and carry
the :data:`FIXED_PERF_CAVEAT` subtitle.
"""

import contextlib
import copy
import io
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..eco_hours import annual_cycle_count, annual_flight_hours
from ..ecomo_economic import EcoMo

FIXED_PERF_CAVEAT = ("performance held fixed; full trend requires "
                     "AWESPA in the loop")

# Subsystems aggregated in the per-run cost breakdown. The operator/
# maintenance crew labour is a separate leaf under the BoS subtree
# (BoS.labour).
SUBSYSTEMS = ('kite', 'tether', 'gStation', 'BoS')


@dataclass(frozen=True)
class Param:
    """A sweepable parameter descriptor.

    Attributes:
        key: Short identifier used in tables and file names.
        domain: Where the parameter lives -- ``'cost'`` (cost inputs
            YAML), ``'settings'`` (economic settings YAML), ``'system'``
            (system YAML) or ``'perf'`` (a cost-side performance knob
            applied to the loaded model in AWESPA-connected mode).
        path: Dotted path within the domain. List indices are written as
            integers, e.g. ``components.kites.0.wing.structure.flat_wing_area``.
            For ``perf`` it is one of ``crest_factor``,
            ``peak_mechanical_power`` or ``rated_power``.
        label: Human-readable axis label.
        unit: Unit string for the axis label.
        low: Default low end of the plausible band.
        high: Default high end of the plausible band.
        fixed_perf: True when the parameter also changes the real
            performance, so plots must carry :data:`FIXED_PERF_CAVEAT`.
    """

    key: str
    domain: str
    path: str
    label: str
    unit: str
    low: float
    high: float
    fixed_perf: bool = False


# Registry of the standard parameters used by the tornado and 1D sweeps.
# Bands are the defaults from the task specification; all are overridable.
PARAMS: Dict[str, Param] = {
    'canopy_life': Param(
        'canopy_life', 'cost',
        'costs.kite.structure.soft.canopy_lifetime_flight_hours',
        'Canopy life', 'h', 100.0, 500.0),
    'tether_oper_life': Param(
        'tether_oper_life', 'cost',
        'costs.tether.operational_life_flight_hours',
        'Tether operational life', 'h', 100.0, 500.0),
    'labour_price': Param(
        'labour_price', 'settings', 'operations.labour_price',
        'Labour price', 'EUR/h', 25.0, 75.0),
    'operating_hours': Param(
        'operating_hours', 'settings', 'operations.operating_hours_per_day',
        'Operating hours', 'h/day', 1.0, 4.0),
    'maintenance_hours': Param(
        'maintenance_hours', 'settings',
        'operations.maintenance_hours_per_flight_hour',
        'Maintenance', 'h/flight-h', 0.05, 0.5),
    'availability': Param(
        'availability', 'settings', 'operations.availability',
        'Availability', '-', 0.3, 1.0),
    'ultracap_price': Param(
        'ultracap_price', 'cost',
        'costs.ground_station.ultracapacitor.price_energy',
        'Ultracapacitor price', 'EUR/kWh', 6000.0, 60000.0),
    'generator_price': Param(
        'generator_price', 'cost',
        'costs.ground_station.generator.power_based.price_power',
        'Generator price', 'EUR/kW', 120.0, 250.0),
    'crest_factor': Param(
        'crest_factor', 'perf', 'crest_factor',
        'Crest factor (peak/rated)', '-', 2.0, 3.5, fixed_perf=True),
    # Winch D/d ratio: sizes the winch drum CAPEX and the tether bending
    # fatigue life. Cost-domain only (no AWESPA re-run).
    'drum_ratio': Param(
        'drum_ratio', 'cost',
        'costs.ground_station.winch.drum_to_tether_diameter_ratio',
        'Drum/tether diameter ratio (D/d)', '-', 20.0, 100.0),
    'canopy_load_exponent': Param(
        'canopy_load_exponent', 'cost',
        'costs.kite.structure.soft.canopy_load_exponent',
        'Canopy load exponent (S-N m)', '-', 0.0, 4.0),
    # Operations automation scales the per-operating-day operating hours
    # with (1 - automation); the operating hours themselves are an
    # operating-pattern assumption swept via 'operating_hours'.
    'automation': Param(
        'automation', 'settings', 'operations.automation',
        'Operations automation', '-', 0.0, 1.0),
    # Design variables for the 2D map (cost at fixed performance)
    'wing_area': Param(
        'wing_area', 'system',
        'components.kites.0.wing.structure.flat_wing_area',
        'Flat wing area', 'm2', 15.0, 60.0, fixed_perf=True),
    'generator_nameplate': Param(
        'generator_nameplate', 'perf', 'peak_mechanical_power',
        'Generator nameplate (peak mech. power)', 'W', 20000.0, 60000.0,
        fixed_perf=True),
}

# The uncertain assumptions shown in the tornado, in registry order: the
# operating/lifetime assumptions that drive the soft-wing prototype LCoE.
TORNADO_KEYS = (
    'availability', 'labour_price', 'maintenance_hours', 'canopy_life',
    'operating_hours', 'automation', 'drum_ratio', 'crest_factor',
)


# ----------------------------------------------------------------------- #
#  YAML / nested-dict helpers
# ----------------------------------------------------------------------- #

def _load_yaml(path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _dump_yaml(data: Dict[str, Any], path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def _coerce_key(key: str):
    """Turn a path segment into an int index when it is numeric."""
    return int(key) if key.lstrip('-').isdigit() else key


def _to_builtin(value: Any) -> Any:
    """Coerce numpy scalars to plain Python so YAML can serialize them."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _set_nested(data: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set a value at a dotted path, descending into dicts and lists."""
    keys = [_coerce_key(k) for k in dotted.split('.')]
    node = data
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def _absolutize_paths(settings: Dict[str, Any], base_dir: Path) -> None:
    """Rewrite every file reference in the settings to an absolute path."""
    inputFiles = settings.get('input_files') or {}
    for key in ('system', 'cost_inputs', 'performance',
                'aep_results', 'power_curves'):
        value = inputFiles.get(key)
        if value:
            inputFiles[key] = str((base_dir / value).resolve())
    windResource = settings.get('wind_resource') or {}
    if windResource.get('resource_file'):
        windResource['resource_file'] = str(
            (base_dir / windResource['resource_file']).resolve())


@contextlib.contextmanager
def _silenced():
    """Suppress the model's stdout chatter and warnings during a sweep."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with contextlib.redirect_stdout(io.StringIO()):
            yield


def _apply_perf_override(model: EcoMo, path: str, value: float) -> None:
    """Apply a cost-side performance override to the loaded model.

    Only valid in AWESPA-connected mode, where the rated and peak powers
    come from ``model.awespaData`` and are read fresh on every
    ``compute_economics`` call.
    """
    if model.awespaData is None:
        raise ValueError(
            "perf overrides (crest factor, peak/rated power) require "
            "AWESPA-connected mode (aep_results + power_curves set).")
    data = model.awespaData
    if path == 'crest_factor':
        data['peakReelOutPower'] = float(value) * data['peRated']
    elif path == 'peak_mechanical_power':
        data['peakReelOutPower'] = float(value)
    elif path == 'rated_power':
        data['peRated'] = float(value)
    else:
        raise ValueError(f"Unknown perf override '{path}'.")


# ----------------------------------------------------------------------- #
#  Per-run breakdown
# ----------------------------------------------------------------------- #

def _walk_capex_opex(node: Any) -> Tuple[float, float]:
    """Sum every CAPEX and OPEX leaf under a results subtree."""
    capex = opex = 0.0
    if isinstance(node, dict):
        capex += node.get('CAPEX', 0.0) or 0.0
        opex += node.get('OPEX', 0.0) or 0.0
        for key, value in node.items():
            if key in ('CAPEX', 'OPEX') or not isinstance(value, dict):
                continue
            subCapex, subOpex = _walk_capex_opex(value)
            capex += subCapex
            opex += subOpex
    return capex, opex


def subsystem_breakdown(eco: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """Per-subsystem (CAPEX, OPEX) totals [EUR, EUR/year]."""
    return {name: (_walk_capex_opex(eco[name]) if name in eco else (0.0, 0.0))
            for name in SUBSYSTEMS}


# Display-level regroup of the subsystem tree for the cost plots (not a
# model change): the crew labour is split out of BoS into its own band.
# The category CAPEX/OPEX sums still equal ICC/OMC.
DISPLAY_CATEGORIES = ('kite', 'tether', 'gstation', 'crew', 'bos')


def display_breakdown(eco: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    """Per-display-category (CAPEX, OPEX) totals [EUR, EUR/year].

    Regroups the raw subsystem tree for presentation:
      - kite     : the kite subtree (structure + KCU/avionics + sensor + ...)
      - tether   : the tether subtree
      - gstation : the ground station 
      - crew     : the operator + maintenance labour (split out of BoS.OM)
      - bos      : the remaining BoS (site prep, foundation, installation,
                   decommissioning, and the per-kW O&M overhead -- no labour)
    """
    kite = _walk_capex_opex(eco.get('kite', {}))
    tether = _walk_capex_opex(eco.get('tether', {}))

    groundStation = eco.get('gStation', {})
    # llsNode = groundStation.get('lls', {})
    # llsCapex = llsNode.get('CAPEX', 0.0) or 0.0
    # llsOpex = llsNode.get('OPEX', 0.0) or 0.0
    gsCapex, gsOpex = _walk_capex_opex(groundStation)
    # gsCapex -= llsCapex
    # gsOpex -= llsOpex

    bos = eco.get('BoS', {})
    # The crew labour is now its own BoS.labour leaf group; pull it out
    # for the 'Ground crew' display band, leaving the per-kW overhead and
    # the BoS CAPEX under 'bos'.
    crewOpex = bos.get('labour', {}).get('OPEX', 0.0) or 0.0
    bosCapex, bosOpex = _walk_capex_opex(bos)
    bosOpex -= crewOpex  # remove labour, leaving the overhead

    return {
        'kite': kite,
        'tether': tether,
        'gstation': (gsCapex, gsOpex),
        # 'lls': (llsCapex, llsOpex),
        'crew': (0.0, crewOpex),
        'bos': (bosCapex, bosOpex),
    }


def make_row(eco: Dict[str, Any], **extra) -> Dict[str, float]:
    """Build one tidy result row from an ``eco`` results dict."""
    metrics = eco['metrics']
    row: Dict[str, float] = dict(extra)
    for key in ('LCoE', 'ICC', 'OMC', 'AEP', 'CF', 'CRF'):
        row[key] = float(metrics[key])
    for name, (capex, opex) in display_breakdown(eco).items():
        row[f'{name}_capex'] = capex
        row[f'{name}_opex'] = opex
    return row


def column(rows: Sequence[Dict[str, float]], key: str) -> np.ndarray:
    """Extract one column from a list of rows as a numpy array."""
    return np.array([row[key] for row in rows], dtype=float)


# ----------------------------------------------------------------------- #
#  Sweep runner
# ----------------------------------------------------------------------- #

class SweepRunner:
    """Runs EcoMo repeatedly with parameter overrides.

    The base configuration is read once; each :meth:`run` deep-copies it,
    applies the requested overrides, materializes temporary YAML files and
    evaluates the model. Use as a context manager so the temporary
    directory is cleaned up::

        with SweepRunner(settings_path) as runner:
            _, eco = runner.run()                       # baseline
            rows = sweep_1d(runner, PARAMS['canopy_life'], values)
    """

    def __init__(self, base_settings_path):
        self.basePath = Path(base_settings_path)
        if not self.basePath.exists():
            raise FileNotFoundError(
                f"Base settings file not found: {self.basePath}")
        self._baseDir = self.basePath.parent
        self._baseSettings = _load_yaml(self.basePath)
        self._tmp = tempfile.TemporaryDirectory(prefix='ecomo_sweep_')
        self.tmpDir = Path(self._tmp.name)

    def __enter__(self) -> 'SweepRunner':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Remove the temporary working directory."""
        self._tmp.cleanup()

    def run(self, overrides: Sequence[Tuple[str, str, Any]] = ()
            ) -> Tuple[EcoMo, Dict[str, Any]]:
        """Evaluate the model once with the given overrides.

        Args:
            overrides: Sequence of ``(domain, path, value)`` tuples, where
                domain is ``'cost'``, ``'settings'``, ``'system'`` or
                ``'perf'``.

        Returns:
            tuple: ``(model, eco)`` -- the evaluated model and its results.
        """
        settings = copy.deepcopy(self._baseSettings)
        _absolutize_paths(settings, self._baseDir)

        byDomain: Dict[str, List[Tuple[str, Any]]] = {
            'cost': [], 'settings': [], 'system': [], 'perf': []}
        for domain, path, value in overrides:
            byDomain[domain].append((path, _to_builtin(value)))

        for path, value in byDomain['settings']:
            _set_nested(settings, path, value)

        if byDomain['cost']:
            cost = _load_yaml(settings['input_files']['cost_inputs'])
            for path, value in byDomain['cost']:
                _set_nested(cost, path, value)
            costPath = self.tmpDir / 'cost_inputs.yml'
            _dump_yaml(cost, costPath)
            settings['input_files']['cost_inputs'] = str(costPath)

        if byDomain['system']:
            system = _load_yaml(settings['input_files']['system'])
            for path, value in byDomain['system']:
                _set_nested(system, path, value)
            systemPath = self.tmpDir / 'system.yml'
            _dump_yaml(system, systemPath)
            settings['input_files']['system'] = str(systemPath)

        settingsPath = self.tmpDir / 'settings.yml'
        _dump_yaml(settings, settingsPath)

        model = EcoMo()
        with _silenced():
            model.load_configuration(settingsPath, validate=False)
            for path, value in byDomain['perf']:
                _apply_perf_override(model, path, value)
            eco = model.compute_economics(validate=False)
        return model, eco


# ----------------------------------------------------------------------- #
#  Sweep front-ends
# ----------------------------------------------------------------------- #

def sweep_1d(runner: SweepRunner, param: Param,
             values: Sequence[float],
             fixed: Sequence[Tuple[str, str, Any]] = ()
             ) -> List[Dict[str, float]]:
    """Run a 1D sweep of one parameter over ``values``.

    Returns a list of tidy rows (one per value) with the swept ``value``,
    the headline metrics and the per-subsystem CAPEX/OPEX breakdown.

    Args:
        runner: Sweep runner.
        param: Parameter to sweep.
        values: Values to evaluate the parameter at.
        fixed: Optional additional overrides applied on every run
            alongside the swept parameter (e.g. to hold automation at 1.0
            while sweeping the maintenance hours, or to enable the
            operational tether-life mode). Same ``(domain, path, value)``
            form as :meth:`SweepRunner.run`.
    """
    fixed = list(fixed)
    rows = []
    for value in values:
        _, eco = runner.run(fixed + [(param.domain, param.path, value)])
        rows.append(make_row(eco, value=float(value)))
    return rows


def sweep_2d(runner: SweepRunner, x_param: Param, x_values: Sequence[float],
             y_param: Param, y_values: Sequence[float]
             ) -> Dict[str, np.ndarray]:
    """Run a 2D sweep over the (x, y) grid.

    Returns a dict with ``'X'``, ``'Y'`` mesh grids and an ``'LCoE'``
    array of shape ``(len(y_values), len(x_values))``.
    """
    lcoe = np.empty((len(y_values), len(x_values)), dtype=float)
    for j, yValue in enumerate(y_values):
        for i, xValue in enumerate(x_values):
            _, eco = runner.run([
                (x_param.domain, x_param.path, xValue),
                (y_param.domain, y_param.path, yValue),
            ])
            lcoe[j, i] = eco['metrics']['LCoE']
    X, Y = np.meshgrid(np.asarray(x_values, dtype=float),
                       np.asarray(y_values, dtype=float))
    return {'X': X, 'Y': Y, 'LCoE': lcoe}


def tornado(runner: SweepRunner, keys: Sequence[str] = TORNADO_KEYS,
            bands: Optional[Dict[str, Tuple[float, float]]] = None
            ) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute a one-at-a-time tornado sensitivity.

    For each parameter the LCoE is evaluated at the low and high end of
    its band (everything else at baseline).

    Args:
        runner: Sweep runner.
        keys: Parameter keys (into :data:`PARAMS`) to include.
        bands: Optional ``{key: (low, high)}`` overrides for the bands.

    Returns:
        tuple: ``(baseline_lcoe, entries)`` where each entry is a dict
        with ``param``, ``low``, ``high`` (band values), ``lcoe_low``,
        ``lcoe_high``, ``swing`` and ``fixed_perf``.
    """
    bands = bands or {}
    _, baseEco = runner.run()
    baseline = float(baseEco['metrics']['LCoE'])

    entries = []
    for key in keys:
        param = PARAMS[key]
        low, high = bands.get(key, (param.low, param.high))
        _, ecoLow = runner.run([(param.domain, param.path, low)])
        _, ecoHigh = runner.run([(param.domain, param.path, high)])
        lcoeLow = float(ecoLow['metrics']['LCoE'])
        lcoeHigh = float(ecoHigh['metrics']['LCoE'])
        entries.append({
            'param': param,
            'low': low, 'high': high,
            'lcoe_low': lcoeLow, 'lcoe_high': lcoeHigh,
            'swing': abs(lcoeHigh - lcoeLow),
            'fixed_perf': param.fixed_perf,
        })
    entries.sort(key=lambda e: e['swing'])
    return baseline, entries


def annual_rates(model: EcoMo) -> Tuple[float, float]:
    """Annual flight hours and pumping-cycle count of a loaded model.

    Used by the tether life-vs-stress diagnostic to convert the
    stress-based fatigue models into replacements per year.
    """
    availability = (model.inputs.operations.availability
                    if model.inputs.operations is not None else 1.0)
    return (annual_flight_hours(model.inputs.performance, availability),
            annual_cycle_count(model.inputs.performance, availability))

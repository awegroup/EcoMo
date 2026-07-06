"""Optional analysis / plotting layer for EcoMo.

A thin, OFF-BY-DEFAULT layer that re-renders the cost breakdown and adds
sweep, sensitivity and diagnostic plots for trend-spotting. It only calls
``compute_economics()`` (directly or via the sweep harness) -- the model
stays a clean inputs->LCoE function and no cost subsystem, metric, or the
existing pie display is modified.

Enable it from the economic settings YAML::

    analysis:
      enabled: true
      plots: [stacked_bar, tornado, sweeps, contour, tether_life, power_curve]

When ``enabled`` is false (the default), nothing here runs and the
existing behaviour is byte-for-byte identical.

Entry points:
    run_analysis(settings_path, outdir, ...): produce all enabled plots.
    maybe_run_analysis(settings_path, outdir, ...): run only if the
        settings' ``analysis`` block is enabled.
"""

import contextlib
import io
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import matplotlib.pyplot as plt

from ...ecomo_economic import EcoMo
from .plots import (
    plot_1d_sweeps,
    plot_2d_contour,
    plot_lcoe_stacked_bar,
    plot_power_curve_wind,
    plot_tether_life_diagnostic,
    plot_tornado,
)
from .sweep import SweepRunner

DEFAULT_PLOTS = ('stacked_bar', 'tornado', 'sweeps', 'contour',
                 'tether_life', 'power_curve')


def analysis_options(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the ``analysis`` block of a settings dict.

    Args:
        settings (dict): Parsed economic settings.

    Returns:
        dict: ``{'enabled': bool, 'plots': tuple, 'n_points': int,
        'grid': tuple, 'tornado_bands': dict|None, 'sweep_keys': seq|None}``.
        Defaults to disabled when the block is absent.
    """
    block = settings.get('analysis') or {}
    grid = block.get('grid', [6, 6])
    return {
        'enabled': bool(block.get('enabled', False)),
        'plots': tuple(block.get('plots') or DEFAULT_PLOTS),
        'n_points': int(block.get('n_points', 7)),
        'grid': (int(grid[0]), int(grid[1])),
        'tornado_bands': block.get('tornado_bands'),
        'tornado_keys': block.get('tornado_keys'),
        'sweep_keys': block.get('sweep_keys'),
    }


def _baseline(settings_path: Path):
    """Load the model and compute the baseline results once, quietly."""
    model = EcoMo()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with contextlib.redirect_stdout(io.StringIO()):
            model.load_configuration(settings_path, validate=False)
            eco = model.compute_economics(validate=False)
    return model, eco


def run_analysis(settings_path, outdir, *,
                 plots=DEFAULT_PLOTS, n_points: int = 7, grid=(6, 6),
                 tornado_bands: Optional[Dict[str, Any]] = None,
                 tornado_keys=None, sweep_keys=None,
                 model: Optional[EcoMo] = None,
                 eco: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Produce the enabled analysis plots for a settings configuration.

    Args:
        settings_path: Path to the economic settings YAML. The sweep
            plots re-run the model from this file.
        outdir: Directory where the PNG figures are written.
        plots: Iterable of plot names to produce (subset of
            :data:`DEFAULT_PLOTS`).
        n_points: Number of points per 1D sweep.
        grid: ``(n_x, n_y)`` resolution of the 2D contour.
        tornado_bands: Optional ``{key: (low, high)}`` band overrides.
        sweep_keys: Optional iterable of parameter keys for the 1D sweeps.
        model, eco: Optional pre-computed baseline (avoids recomputing the
            baseline; the sweep plots still re-run from ``settings_path``).

    Returns:
        dict: Mapping of plot name to the produced PNG path(s).
    """
    settings_path = Path(settings_path)
    outdir = Path(outdir)
    plots = set(plots)

    if model is None or eco is None:
        model, eco = _baseline(settings_path)

    produced: Dict[str, Any] = {}
    with SweepRunner(settings_path) as runner:
        if 'stacked_bar' in plots:
            produced['stacked_bar'] = plot_lcoe_stacked_bar(eco, outdir)
        if 'tether_life' in plots:
            produced['tether_life'] = plot_tether_life_diagnostic(
                model, eco, outdir)
        if 'power_curve' in plots:
            produced['power_curve'] = plot_power_curve_wind(model, eco, outdir)
        if 'tornado' in plots:
            produced['tornado'] = plot_tornado(
                runner, outdir, keys=tornado_keys, bands=tornado_bands)
        if 'sweeps' in plots:
            keys = sweep_keys or ('crest_factor', 'canopy_life',
                                  'tether_oper_life', 'availability',
                                  'labour_price')
            produced['sweeps'] = plot_1d_sweeps(
                runner, outdir, keys=keys, n_points=n_points)
        if 'contour' in plots:
            produced['contour'] = plot_2d_contour(
                runner, outdir, n_x=grid[0], n_y=grid[1])
    plt.close('all')
    return produced


def maybe_run_analysis(settings_path, outdir, *,
                       model: Optional[EcoMo] = None,
                       eco: Optional[Dict[str, Any]] = None
                       ) -> Optional[Dict[str, Any]]:
    """Run the analysis only if the settings enable it.

    Reads the ``analysis`` block from ``settings_path``; returns None
    (running nothing) when it is absent or disabled.
    """
    settings_path = Path(settings_path)
    with open(settings_path, 'r', encoding='utf-8') as f:
        settings = yaml.safe_load(f)
    options = analysis_options(settings)
    if not options['enabled']:
        return None
    return run_analysis(
        settings_path, outdir,
        plots=options['plots'], n_points=options['n_points'],
        grid=options['grid'], tornado_bands=options['tornado_bands'],
        tornado_keys=options['tornado_keys'],
        sweep_keys=options['sweep_keys'], model=model, eco=eco)


__all__ = ['run_analysis', 'maybe_run_analysis', 'analysis_options',
           'DEFAULT_PLOTS']

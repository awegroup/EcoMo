"""Analysis plots for EcoMo (optional, off by default).

A thin presentation layer over :mod:`ecomo.ecomo.analysis.sweep`. Every
function only reads results from ``compute_economics()`` (directly or via
the sweep harness) and renders a matplotlib figure saved as a PNG. None of
the cost subsystems, the metrics, or the existing pie display
(``eco_display_results``) are touched.

The cost-side caveat from the sweep harness applies: plots built on
parameters that would also change the real performance carry the
:data:`~ecomo.ecomo.analysis.sweep.FIXED_PERF_CAVEAT` subtitle.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# The analysis figures are always saved (never shown), so the active
# matplotlib backend is left untouched -- forcing 'Agg' here would turn the
# run script's interactive pie display into a silent no-op. Headless callers
# (tests, CI) select a non-interactive backend via MPLBACKEND/matplotlib.use.
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..constants import PA_PER_GPA
from ..eco_costs import TetherCosts
from ..loaders.eco_load_cost_inputs import _resolve_bending_life_a1

# Drum-to-tether diameter ratios (D/d) drawn on the tether life-vs-stress
# diagnostic (figure 5) to show how the winch drum choice shifts the
# bending-fatigue curve. Each gets its own line style; a1 is resolved from
# the cost file's bending_life_a1_table at each ratio.
_DRUM_RATIOS_DIAGNOSTIC = (10, 20, 30, 100)
_DRUM_RATIO_LINESTYLES = ('-', '--', '-.', ':')
from .sweep import (
    DISPLAY_CATEGORIES,
    FIXED_PERF_CAVEAT,
    PARAMS,
    Param,
    SweepRunner,
    annual_rates,
    column,
    display_breakdown,
    sweep_1d,
    sweep_2d,
    tornado,
)

# Display categories used by all cost plots (Ground crew split out of BoS,
# Launch & land split out of the ground station). Stable colour and label
# per category; Ground crew is red to flag it as a driving subsystem.
_CATEGORIES = DISPLAY_CATEGORIES
_SUBSYSTEM_COLORS = {
    'kite': '#1f77b4',       # blue
    'tether': '#ff7f0e',     # orange
    'gstation': '#2ca02c',   # green
    'lls': '#9467bd',        # purple
    'crew': '#d62728',       # red (driving subsystem)
    'bos': '#9467bd',        # purple
}
_PRETTY = {
    'kite': 'Kite', 'tether': 'Tether', 'gstation': 'Ground station',
    'lls': 'Launch & land', 'crew': 'Ground crew', 'bos': 'BoS',
}


def _save(fig, outdir: Path, name: str) -> Path:
    """Save a figure as a PNG in ``outdir`` and return its path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return path


def _lcoe_contributions(eco: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Per-display-category LCoE contributions [EUR/MWh].

    CAPEX contributes its annuity ``CAPEX * CRF / AEP`` and OPEX
    contributes ``OPEX / AEP`` -- together these sum to the LCoE.
    """
    metrics = eco['metrics']
    crf, aep = metrics['CRF'], metrics['AEP']
    out = {}
    for name, (capex, opex) in display_breakdown(eco).items():
        out[name] = {'capex': capex * crf / aep, 'opex': opex / aep}
    return out


# ----------------------------------------------------------------------- #
#  1. Cost breakdown: CAPEX, OPEX and LCoE by subsystem (absolute values)
# ----------------------------------------------------------------------- #

def _stack_bar(ax, values_per_subsystem, hatched=False, contrib=None):
    """Draw one stacked bar (per-subsystem) on ``ax``; return top value.

    When ``contrib`` is given (the LCoE panel), each subsystem is split
    into a hatched CAPEX-annuity segment and a solid OPEX segment;
    otherwise a single solid segment per subsystem is drawn from
    ``values_per_subsystem``.
    """
    bottom = 0.0
    for name in _CATEGORIES:
        color = _SUBSYSTEM_COLORS[name]
        if contrib is not None:
            for kind, hatch in (('capex', '////'), ('opex', None)):
                value = contrib[name][kind]
                if value <= 0:
                    continue
                ax.bar(0, value, bottom=bottom, width=0.6, color=color,
                       hatch=hatch, edgecolor='white')
                bottom += value
        else:
            value = values_per_subsystem.get(name, 0.0)
            if value <= 0:
                continue
            ax.bar(0, value, bottom=bottom, width=0.6, color=color,
                   edgecolor='white', label=_PRETTY[name])
            bottom += value
    ax.set_xticks([])
    ax.set_xlim(-0.6, 0.6)
    return bottom


def plot_lcoe_stacked_bar(eco: Dict[str, Any], outdir: Path) -> Path:
    """Three-panel cost breakdown by subsystem, in absolute values.

    Panel 1 -- CAPEX [EUR]; panel 2 -- OPEX [EUR/year]; panel 3 -- LCoE
    contribution [EUR/MWh] (CAPEX annuity hatched, OPEX solid). All axes
    are absolute (no percentages). Shows both where the up-front money
    goes (CAPEX), where the recurring money goes (OPEX), and how the two
    combine into the LCoE via the capital recovery factor.
    """
    metrics = eco['metrics']
    lcoe, icc, omc = metrics['LCoE'], metrics['ICC'], metrics['OMC']
    breakdown = display_breakdown(eco)              # {name: (capex, opex)}
    capexPer = {name: cap for name, (cap, _) in breakdown.items()}
    opexPer = {name: op for name, (_, op) in breakdown.items()}
    contrib = _lcoe_contributions(eco)              # {name: {capex, opex}}

    fig, (axCap, axOpex, axLcoe) = plt.subplots(1, 3, figsize=(13, 6.5))

    _stack_bar(axCap, capexPer)
    axCap.set_ylabel('CAPEX [EUR]')
    axCap.set_title(f'CAPEX\n(ICC = {icc:,.0f} EUR)')

    _stack_bar(axOpex, opexPer)
    axOpex.set_ylabel('OPEX [EUR/year]')
    axOpex.set_title(f'OPEX\n(OMC = {omc:,.0f} EUR/yr)')

    _stack_bar(axLcoe, None, contrib=contrib)
    axLcoe.set_ylabel('LCoE contribution [EUR/MWh]')
    axLcoe.set_title(f'LCoE\n({lcoe:.0f} EUR/MWh)')

    # Shared category legend built explicitly from all display categories
    # (not from the CAPEX panel handles -- Ground crew cost has no CAPEX and
    # would otherwise be dropped) plus a note that the LCoE panel splits each
    # category into CAPEX annuity (hatched) and OPEX (solid).
    handles = [Patch(facecolor=_SUBSYSTEM_COLORS[name], label=_PRETTY[name])
               for name in _CATEGORIES]
    labels = [_PRETTY[name] for name in _CATEGORIES]
    legend_note = (
        Patch(facecolor='white', edgecolor='dimgray', hatch='////'),
        Patch(facecolor='white', edgecolor='dimgray')
    )
    fig.legend(
        handles + [legend_note],
        labels + ['LCoE panel: hatched = CAPEX annuity, solid = OPEX'],
        loc='lower center',
        ncol=len(labels) + 1,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    fig.suptitle('Cost breakdown by category: CAPEX, OPEX and LCoE',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 0.97, 0.95])
    return _save(fig, outdir, '1_cost_breakdown')


# ----------------------------------------------------------------------- #
#  2. Tornado / sensitivity chart
# ----------------------------------------------------------------------- #

def plot_tornado(runner: SweepRunner, outdir: Path,
                 keys: Optional[Sequence[str]] = None,
                 bands: Optional[Dict[str, Any]] = None) -> Path:
    """Horizontal tornado of the LCoE swing per uncertain assumption."""
    from .sweep import TORNADO_KEYS
    keys = keys or TORNADO_KEYS
    baseline, entries = tornado(runner, keys, bands)

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(entries) + 2))
    for i, entry in enumerate(entries):
        lo, hi = entry['lcoe_low'], entry['lcoe_high']
        ax.barh(i, lo - baseline, left=baseline, height=0.7,
                color='#4C72B0', alpha=0.85)
        ax.barh(i, hi - baseline, left=baseline, height=0.7,
                color='#C44E52', alpha=0.85)
        param: Param = entry['param']
        note = ' *' if entry['fixed_perf'] else ''
        ax.text(baseline, i, f"  {param.label}{note}  ",
                va='center', ha='left' if hi >= baseline else 'right',
                fontsize=8)

    ax.axvline(baseline, color='black', lw=1.5)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([
        f"{e['param'].low:g}-{e['param'].high:g} {e['param'].unit}"
        if (e['param'].low, e['param'].high) == (e['low'], e['high'])
        else f"{e['low']:g}-{e['high']:g} {e['param'].unit}"
        for e in entries], fontsize=8)
    ax.set_xlabel('LCoE [EUR/MWh]')
    ax.set_title(f'Tornado sensitivity  (baseline LCoE = {baseline:.0f} '
                 f'EUR/MWh)', pad=24)
    ax.text(0.5, 1.01, 'blue = low end, red = high end of band   '
            '(*  cost at fixed performance)', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8, color='dimgray')
    fig.tight_layout()
    return _save(fig, outdir, '2_tornado')


# ----------------------------------------------------------------------- #
#  3. 1D design / assumption sweeps
# ----------------------------------------------------------------------- #

def plot_1d_sweep(runner: SweepRunner, param: Param, outdir: Path,
                  n_points: int = 7,
                  values: Optional[Sequence[float]] = None,
                  fixed: Sequence = (),
                  name: Optional[str] = None,
                  title: Optional[str] = None,
                  logx: bool = False,
                  markers: Optional[Dict[float, str]] = None) -> Path:
    """LCoE vs one parameter, with the cost breakdown as a stacked area.

    Args:
        runner: Sweep runner.
        param: Parameter to sweep.
        outdir: Output directory.
        n_points: Points when ``values`` is not given (linspace over the
            parameter band).
        values: Explicit values to sweep (overrides the default band).
        fixed: Additional overrides held constant across the sweep (e.g.
            automation = 1.0), passed through to :func:`sweep_1d`.
        name: Output file stem (without extension); defaults to
            ``3_sweep_{param.key}``.
        title: Plot title; defaults to ``LCoE vs {param.label}``.
        logx: Use a log x-axis (for wide-range life sweeps).
        markers: Optional ``{x_value: label}`` vertical annotations.
    """
    if values is None:
        values = np.linspace(param.low, param.high, n_points)
    values = np.asarray(values, dtype=float)
    rows = sweep_1d(runner, param, values, fixed=fixed)
    x = column(rows, 'value')
    lcoe = column(rows, 'LCoE')
    crf = column(rows, 'CRF')
    aep = column(rows, 'AEP')

    fig, (axTop, axBot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axTop.plot(x, lcoe, 'o-', color='black', lw=2)
    axTop.set_ylabel('LCoE [EUR/MWh]')
    axTop.grid(True, alpha=0.3)
    axTop.set_title(title or f'LCoE vs {param.label}')

    # Stacked area of per-category LCoE contribution (capex annuity + opex)
    stack, labels, colors = [], [], []
    for category in _CATEGORIES:
        contribution = (column(rows, f'{category}_capex') * crf / aep +
                        column(rows, f'{category}_opex') / aep)
        if np.any(contribution > 0):
            stack.append(contribution)
            labels.append(_PRETTY[category])
            colors.append(_SUBSYSTEM_COLORS[category])
    axBot.stackplot(x, *stack, labels=labels, colors=colors, alpha=0.85)
    axBot.set_ylabel('LCoE contribution [EUR/MWh]')
    axBot.set_xlabel(f'{param.label} [{param.unit}]')
    axBot.legend(loc='upper right', fontsize=8)
    axBot.grid(True, alpha=0.3)

    if logx:
        axTop.set_xscale('log')
        axBot.set_xscale('log')
    for ax in (axTop, axBot):
        for xValue, markLabel in (markers or {}).items():
            ax.axvline(xValue, color='gray', ls=':', lw=1.2)
        if markers and ax is axTop:
            for xValue, markLabel in markers.items():
                ax.annotate(markLabel, xy=(xValue, ax.get_ylim()[1]),
                            xytext=(2, -8), textcoords='offset points',
                            fontsize=8, va='top', color='dimgray')

    if param.fixed_perf:
        fig.text(0.5, 0.005, FIXED_PERF_CAVEAT, ha='center', fontsize=8,
                 style='italic', color='dimgray')
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    return _save(fig, outdir, name or f'3_sweep_{param.key}')


def plot_1d_sweeps(runner: SweepRunner, outdir: Path,
                   keys: Sequence[str] = ('canopy_life', 'availability',
                                          'labour_price'),
                   n_points: int = 7) -> List[Path]:
    """Render a set of standard keyed 1D sweeps (one figure each).

    Used when explicit ``sweep_keys`` are requested. The default analysis
    run instead calls :func:`plot_requested_sweeps` for the curated
    labour/lifetime sweep set.
    """
    return [plot_1d_sweep(runner, PARAMS[key], outdir, n_points)
            for key in keys]


def _sweep_specs(n_points: int) -> List[Dict[str, Any]]:
    """The six curated labour/lifetime/availability sweeps (shared by the
    individual-figure and the combined-overview renderers)."""
    MANUAL = ('settings', 'operations.automation', 0.0)
    AUTOMATED = ('settings', 'operations.automation', 1.0)
    OPER_ON = ('cost', 'costs.tether.operational_life_enabled', True)
    # 1 h/day, 1 h/week, 1 h/month expressed in h/day
    perDay = {'1/day': 1.0, '1/week': 1.0 / 7.0, '1/month': 1.0 / 30.0}
    return [
        dict(param=PARAMS['maintenance_hours'],
             values=np.linspace(0.0, 3.0, n_points), fixed=[MANUAL],
             name='3a_sweep_maintenance_hours_manual',
             title='Maintenance hours (manual, automation = 0)'),
        dict(param=PARAMS['automation'],
             values=np.linspace(0.0, 1.0, n_points), fixed=[],
             name='3b_sweep_automation',
             title='Operations automation'),
        dict(param=PARAMS['maintenance_hours'],
             values=np.linspace(perDay['1/month'], perDay['1/day'], n_points),
             fixed=[AUTOMATED], name='3c_sweep_maintenance_hours_automated',
             title='Maintenance hours (fully automated, automation = 1)',
             markers={v: k for k, v in perDay.items()}),
        dict(param=PARAMS['tether_oper_life'],
             values=np.geomspace(250.0, 20000.0, n_points), fixed=[OPER_ON],
             name='3d_sweep_tether_oper_life',
             title='Tether operational life (enabled) vs bending limit',
             logx=True),
        dict(param=PARAMS['canopy_life'],
             values=np.linspace(500.0, 5000.0, n_points), fixed=[],
             name='3e_sweep_canopy_life',
             title='Kite (canopy) lifetime'),
        dict(param=PARAMS['availability'],
             values=np.linspace(0.8, 1.0, n_points), fixed=[],
             name='3f_sweep_availability',
             title='Availability'),
    ]


def plot_requested_sweeps(runner: SweepRunner, outdir: Path,
                          n_points: int = 7) -> List[Path]:
    """The curated sweeps, one figure each (see :func:`_sweep_specs`)."""
    return [plot_1d_sweep(runner, s['param'], outdir, values=s['values'],
                          fixed=s['fixed'], name=s['name'], title=s['title'],
                          logx=s.get('logx', False), markers=s.get('markers'))
            for s in _sweep_specs(n_points)]


def plot_sweep_overview(runner: SweepRunner, outdir: Path,
                        n_points: int = 7) -> Path:
    """All six curated sweeps in one figure (2x3 grid of subplots).

    Each subplot is a stacked area of the per-category LCoE contributions
    over the swept parameter, with the total LCoE as the black top line.
    One shared category legend for the whole figure.
    """
    specs = _sweep_specs(n_points)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    usedCategories: List[str] = []
    for ax, spec in zip(axes.flat, specs):
        rows = sweep_1d(runner, spec['param'], spec['values'],
                        fixed=spec['fixed'])
        x = column(rows, 'value')
        crf, aep, lcoe = (column(rows, 'CRF'), column(rows, 'AEP'),
                          column(rows, 'LCoE'))
        stack, colors = [], []
        for category in _CATEGORIES:
            contribution = (column(rows, f'{category}_capex') * crf / aep +
                            column(rows, f'{category}_opex') / aep)
            if np.any(contribution > 0):
                stack.append(contribution)
                colors.append(_SUBSYSTEM_COLORS[category])
                if category not in usedCategories:
                    usedCategories.append(category)
        ax.stackplot(x, *stack, colors=colors, alpha=0.85)
        ax.plot(x, lcoe, color='black', lw=1.5)          # total LCoE
        ax.set_title(spec['title'], fontsize=10)
        ax.set_xlabel(f"{spec['param'].label} [{spec['param'].unit}]",
                      fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)
        if spec.get('logx'):
            ax.set_xscale('log')
        for xValue, markLabel in (spec.get('markers') or {}).items():
            ax.axvline(xValue, color='gray', ls=':', lw=1.0)
            ax.annotate(markLabel, xy=(xValue, ax.get_ylim()[1]),
                        xytext=(2, -8), textcoords='offset points',
                        fontsize=7.5, va='top', color='dimgray')
    for ax in axes[:, 0]:
        ax.set_ylabel('LCoE [EUR/MWh]', fontsize=9)

    handles = [Patch(facecolor=_SUBSYSTEM_COLORS[c], label=_PRETTY[c])
               for c in _CATEGORIES if c in usedCategories]
    handles.append(Line2D([0], [0], color='black', lw=1.5, label='Total LCoE'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('LCoE sensitivity sweeps', fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return _save(fig, outdir, '3_sweep_overview')


# ----------------------------------------------------------------------- #
#  4. 2D contour / heatmap
# ----------------------------------------------------------------------- #

def plot_2d_contour(runner: SweepRunner, outdir: Path,
                    x_key: str = 'wing_area',
                    y_key: str = 'generator_nameplate',
                    n_x: int = 6, n_y: int = 6) -> Path:
    """LCoE over a 2D grid of (wing area, generator nameplate)."""
    xParam, yParam = PARAMS[x_key], PARAMS[y_key]
    xValues = np.linspace(xParam.low, xParam.high, n_x)
    yValues = np.linspace(yParam.low, yParam.high, n_y)
    grid = sweep_2d(runner, xParam, xValues, yParam, yValues)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    filled = ax.contourf(grid['X'], grid['Y'], grid['LCoE'], levels=15,
                         cmap='viridis')
    lines = ax.contour(grid['X'], grid['Y'], grid['LCoE'], levels=8,
                       colors='white', linewidths=0.6)
    ax.clabel(lines, inline=True, fontsize=7, fmt='%.0f')
    fig.colorbar(filled, ax=ax, label='LCoE [EUR/MWh]')
    ax.set_xlabel(f'{xParam.label} [{xParam.unit}]')
    ax.set_ylabel(f'{yParam.label} [{yParam.unit}]')
    ax.set_title('LCoE over wing area and generator nameplate')
    fig.text(0.5, 0.005, f'cost at fixed performance ({FIXED_PERF_CAVEAT})',
             ha='center', fontsize=8, style='italic', color='dimgray')
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    return _save(fig, outdir, '4_lcoe_contour')


# ----------------------------------------------------------------------- #
#  5. Tether life-vs-stress diagnostic
# ----------------------------------------------------------------------- #

def _bending_a1_table(model) -> Optional[Dict[str, Any]]:
    """Read the bending_life_a1_table from the model's cost YAML, or None."""
    try:
        import yaml
        with open(model.costInputsPath, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        return raw['costs']['tether'].get('bending_life_a1_table')
    except (OSError, KeyError, TypeError, AttributeError):
        return None


def plot_tether_life_diagnostic(model, eco: Dict[str, Any],
                                outdir: Path) -> Path:
    """Tether replacement frequency vs fibre stress.

    Creep (green, creep polynomial) is stress-driven only. Bending (blue,
    10^(a1-a2*sigma)) is drawn as a family over the winch drum-to-tether
    diameter ratios D/d (each a line style): a1 is resolved from the cost
    file's calibrated bending_life_a1_table at each ratio, so a smaller
    drum (tighter bend, lower a1) raises the bending replacement frequency
    (shortens life). When no table is configured a single bending curve is
    drawn instead. The empirical operational-wear life (if the toggle is
    on) is a flat red line. A marker shows the V3's operating stress.
    """
    costs: TetherCosts = model.costs.tether
    flightHours, cycleCount = annual_rates(model)

    # Focus on the soft-wing operating band (V3 sits at ~0.3 GPa); the
    # allowable-stress design sweep runs 0.3-0.5 GPa.
    sigmaMin, sigmaMax = 0.2, 0.8                         # [GPa]
    sigma = np.linspace(sigmaMin, sigmaMax, 250)          # [GPa]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Creep frequency (stress-driven only, independent of D/d): the model
    # treats 10^poly(sigma) as a life in years, so the frequency is its
    # reciprocal (no cycle-rate multiplication).
    freqCreep = None
    if costs.creepLifeCoefficients is not None:
        freqCreep = 1.0 / (10.0 ** np.polyval(
            costs.creepLifeCoefficients, sigma))

    # Bending family over D/d (blue, one line style each), plus a single
    # creep curve (green): creep is stress-driven only and does not depend
    # on D/d. Falls back to a single bending curve when no a1(D/d) table
    # is available.
    a1Table = _bending_a1_table(model)
    if a1Table is not None and costs.bendingLifeA2 is not None:
        for ratio, lineStyle in zip(_DRUM_RATIOS_DIAGNOSTIC,
                                    _DRUM_RATIO_LINESTYLES):
            a1 = _resolve_bending_life_a1(
                {'bending_life_a1_table': a1Table}, ratio)
            nFail = 10.0 ** (a1 - costs.bendingLifeA2 * sigma)
            freq = (costs.nBends * cycleCount) / nFail
            ax.plot(sigma, freq, color='#4C72B0', ls=lineStyle,
                    label=f'Bending, D/d = {ratio:g} (a1 = {a1:.2f})')
    elif costs.bendingLifeA1 is not None:
        nFail = 10.0 ** (costs.bendingLifeA1 - costs.bendingLifeA2 * sigma)
        ax.plot(sigma, (costs.nBends * cycleCount) / nFail,
                label='Bending fatigue', color='#4C72B0')

    if freqCreep is not None:
        ax.plot(sigma, freqCreep, label='Creep rupture (independent of D/d)',
                color='#55A868', lw=2.0)

    if costs.operationalLife is not None:
        freqOper = np.full_like(sigma, flightHours / costs.operationalLife)
        ax.plot(sigma, freqOper, label='Operational wear (empirical)',
                color='#C44E52', ls='--')

    # Marker at the V3 operating stress (AEP-weighted average)
    if 'tether' in eco and 'sigma' in eco['tether']:
        perf = model.inputs.performance
        weight = perf.windPdf * (np.asarray(perf.averagePower) > 0)
        sigmaArr = np.asarray(eco['tether']['sigma'])
        if weight.sum() > 0:
            sigmaOp = float(np.average(sigmaArr, weights=weight)) / PA_PER_GPA
            ax.axvline(sigmaOp, color='gray', ls=':', lw=1.5)
            ax.annotate(f'V3 operating\nstress ~ {sigmaOp:.2f} GPa',
                        xy=(sigmaOp, ax.get_ylim()[1]),
                        xytext=(5, -10), textcoords='offset points',
                        fontsize=8, va='top')

    ax.set_yscale('log')
    ax.set_xlim(sigmaMin, sigmaMax)
    ax.set_xlabel('Fibre stress [GPa]')
    ax.set_ylabel('Replacement frequency [1/year]')
    hasFamily = a1Table is not None and costs.bendingLifeA2 is not None
    if hasFamily:
        ax.set_title('Tether life vs stress: bending fatigue per winch drum '
                     'ratio D/d, vs creep')
    elif costs.operationalLife is not None:
        ax.set_title('Tether life vs stress: fatigue governs high stress, '
                     'operational wear low stress')
    else:
        ax.set_title('Tether life vs stress: bending/creep fatigue only '
                     '(operational-wear mode off)')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.text(0.5, 0.005, 'fatigue modes converted to 1/year using the '
             'V3 annual cycle/flight rates', ha='center', fontsize=8,
             style='italic', color='dimgray')
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    return _save(fig, outdir, '5_tether_life_vs_stress')


# ----------------------------------------------------------------------- #
#  6. Power curve + wind distribution sanity panel
# ----------------------------------------------------------------------- #

def plot_power_curve_wind(model, eco: Dict[str, Any], outdir: Path) -> Path:
    """Two-panel power-curve / wind-resource / energy-yield summary.

    (a) The power curve: average electrical (cycle) power vs wind speed,
        with the rated power marked. Cleanly separates the machine's
        electrical output from the (much higher) peak mechanical reel-out
        power, which is noted as text rather than mixed onto the curve.
    (b) Where the energy comes from: the wind resource (PDF) and the
        energy-yield density (power x PDF, whose area is the AEP), both
        normalized to their peak so the shift is visible -- energy is
        produced at markedly higher winds than the most probable wind.
        The AEP-weighted mean wind speed is marked.
    """
    perf = model.inputs.performance
    metrics = eco['metrics']
    wind = np.asarray(perf.windSpeeds)
    avgKw = np.asarray(perf.averagePower) / 1e3
    pdf = np.asarray(perf.windPdf, dtype=float)
    energy = avgKw * pdf                                   # energy-yield density

    fig, (axP, axE) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True)

    # -- (a) power curve ---------------------------------------------------
    axP.plot(wind, avgKw, 'o-', color='#4C72B0', lw=2,
             label='Average electrical (cycle) power')
    axP.axhline(perf.ratedPower / 1e3, color='#C44E52', ls='--',
                label=f'Rated power ({perf.ratedPower / 1e3:.1f} kW)')
    axP.set_ylabel('Power [kW]')
    axP.set_ylim(bottom=0)
    axP.grid(True, alpha=0.3)
    axP.legend(fontsize=9, loc='lower right')
    axP.set_title('(a)  Power curve', fontsize=11, loc='left')
    if perf.peakMechanicalPower:
        axP.text(0.02, 0.95,
                 f'Peak mechanical reel-out power: '
                 f'{perf.peakMechanicalPower / 1e3:.0f} kW '
                 f'(sizes the drivetrain, not the electrical output)',
                 transform=axP.transAxes, fontsize=8, va='top',
                 color='dimgray', style='italic')

    # -- (b) wind resource vs energy yield (normalized shapes) -------------
    pdfNorm = pdf / pdf.max() if pdf.max() > 0 else pdf
    energyNorm = energy / energy.max() if energy.max() > 0 else energy
    axE.fill_between(wind, pdfNorm, color='gray', alpha=0.30,
                     label='Wind resource (PDF)')
    axE.fill_between(wind, energyNorm, color='#4C72B0', alpha=0.35,
                     label='Energy yield  (power x PDF)  → area = AEP')
    axE.plot(wind, energyNorm, color='#4C72B0', lw=1.5)

    if energy.sum() > 0:
        vMeanEnergy = float(np.sum(wind * energy) / np.sum(energy))
        axE.axvline(vMeanEnergy, color='#C44E52', ls=':', lw=1.6)
        axE.annotate(f'AEP-weighted\nmean {vMeanEnergy:.1f} m/s',
                     xy=(vMeanEnergy, 1.0), xytext=(5, -4),
                     textcoords='offset points', fontsize=8, va='top',
                     color='#C44E52')
    if pdf.sum() > 0:
        vMeanWind = float(np.sum(wind * pdf) / np.sum(pdf))
        axE.axvline(vMeanWind, color='dimgray', ls=':', lw=1.2)
        axE.annotate(f'mean wind {vMeanWind:.1f} m/s',
                     xy=(vMeanWind, 0.55), xytext=(-5, 0),
                     textcoords='offset points', fontsize=8, va='center',
                     ha='right', color='dimgray')

    axE.set_xlabel('Wind speed [m/s]')
    axE.set_ylabel('Normalized density [-]')
    axE.set_ylim(0, 1.12)
    axE.grid(True, alpha=0.3)
    axE.legend(fontsize=9, loc='upper right')
    axE.set_title('(b)  Where the energy comes from', fontsize=11, loc='left')

    fig.suptitle(f'Power curve and wind resource  '
                 f'(CF = {metrics["CF"]:.2f}, AEP = {metrics["AEP"]:.1f} '
                 f'MWh/yr)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, outdir, '6_power_curve_wind')

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

from ..constants import BEND_LIFE_CORRECTION, PA_PER_GPA
from ..eco_costs import TetherCosts
from .sweep import (
    FIXED_PERF_CAVEAT,
    PARAMS,
    SUBSYSTEMS,
    Param,
    SweepRunner,
    annual_rates,
    column,
    subsystem_breakdown,
    sweep_1d,
    sweep_2d,
    tornado,
)

# Stable colour per subsystem across all figures
_SUBSYSTEM_COLORS = {
    name: color for name, color in zip(
        SUBSYSTEMS, plt.get_cmap('tab10').colors)
}
_PRETTY = {
    'kite': 'Kite', 'tether': 'Tether', 'gStation': 'Ground station',
    'BoS': 'Balance of system', 'operations': 'Operations (labour)',
}


def _save(fig, outdir: Path, name: str) -> Path:
    """Save a figure as a PNG in ``outdir`` and return its path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return path


def _lcoe_contributions(eco: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Per-subsystem LCoE contributions [EUR/MWh].

    CAPEX contributes its annuity ``CAPEX * CRF / AEP`` and OPEX
    contributes ``OPEX / AEP`` -- together these sum to the LCoE.
    """
    metrics = eco['metrics']
    crf, aep = metrics['CRF'], metrics['AEP']
    out = {}
    for name, (capex, opex) in subsystem_breakdown(eco).items():
        out[name] = {'capex': capex * crf / aep, 'opex': opex / aep}
    return out


# ----------------------------------------------------------------------- #
#  1. Stacked-bar LCoE decomposition (the clearer pie replacement)
# ----------------------------------------------------------------------- #

def plot_lcoe_stacked_bar(eco: Dict[str, Any], outdir: Path) -> Path:
    """Stacked-bar decomposition of the LCoE by subsystem and CAPEX/OPEX.

    Two panels: absolute EUR/MWh and percentage share. CAPEX-annuity
    segments are hatched, OPEX segments solid. Shows at a glance that OPEX
    (labour, kite/tether replacement) dominates and CAPEX is thin.
    """
    contrib = _lcoe_contributions(eco)
    lcoe = eco['metrics']['LCoE']

    fig, (axAbs, axPct) = plt.subplots(1, 2, figsize=(11, 6))
    for ax, normalize in ((axAbs, False), (axPct, True)):
        bottom = 0.0
        scale = 100.0 / lcoe if normalize else 1.0
        for name in SUBSYSTEMS:
            color = _SUBSYSTEM_COLORS[name]
            for kind, hatch in (('capex', '////'), ('opex', None)):
                value = contrib[name][kind] * scale
                if value <= 0:
                    continue
                label = (f'{_PRETTY[name]} '
                         f'{"CAPEX annuity" if kind == "capex" else "OPEX"}')
                ax.bar(0, value, bottom=bottom, width=0.6, color=color,
                       hatch=hatch, edgecolor='white', label=label)
                bottom += value
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
    axAbs.set_ylabel('LCoE contribution [EUR/MWh]')
    axAbs.set_title(f'Absolute  (total LCoE = {lcoe:.0f} EUR/MWh)')
    axPct.set_ylabel('Share of LCoE [%]')
    axPct.set_title('Percentage share')
    axPct.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8,
                 title='Hatched = CAPEX annuity')
    fig.suptitle('LCoE decomposition by subsystem', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, outdir, '1_lcoe_stacked_bar')


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
                  n_points: int = 7) -> Path:
    """LCoE vs one parameter, with the cost breakdown as a stacked area."""
    values = np.linspace(param.low, param.high, n_points)
    rows = sweep_1d(runner, param, values)
    x = column(rows, 'value')
    lcoe = column(rows, 'LCoE')
    crf = column(rows, 'CRF')
    aep = column(rows, 'AEP')

    fig, (axTop, axBot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axTop.plot(x, lcoe, 'o-', color='black', lw=2)
    axTop.set_ylabel('LCoE [EUR/MWh]')
    axTop.grid(True, alpha=0.3)
    axTop.set_title(f'LCoE vs {param.label}')

    # Stacked area of per-subsystem LCoE contribution (capex annuity + opex)
    stack = []
    labels = []
    colors = []
    for name in SUBSYSTEMS:
        contribution = (column(rows, f'{name}_capex') * crf / aep +
                        column(rows, f'{name}_opex') / aep)
        if np.any(contribution > 0):
            stack.append(contribution)
            labels.append(_PRETTY[name])
            colors.append(_SUBSYSTEM_COLORS[name])
    axBot.stackplot(x, *stack, labels=labels, colors=colors, alpha=0.85)
    axBot.set_ylabel('LCoE contribution [EUR/MWh]')
    axBot.set_xlabel(f'{param.label} [{param.unit}]')
    axBot.legend(loc='upper right', fontsize=8)
    axBot.grid(True, alpha=0.3)

    if param.fixed_perf:
        fig.text(0.5, 0.005, FIXED_PERF_CAVEAT, ha='center', fontsize=8,
                 style='italic', color='dimgray')
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    return _save(fig, outdir, f'3_sweep_{param.key}')


def plot_1d_sweeps(runner: SweepRunner, outdir: Path,
                   keys: Sequence[str] = ('crest_factor', 'canopy_life',
                                          'tether_oper_life', 'availability',
                                          'labour_price'),
                   n_points: int = 7) -> List[Path]:
    """Render the standard set of 1D sweeps, one figure each."""
    return [plot_1d_sweep(runner, PARAMS[key], outdir, n_points)
            for key in keys]


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

def plot_tether_life_diagnostic(model, eco: Dict[str, Any],
                                outdir: Path) -> Path:
    """Tether replacement frequency vs fibre stress for the three modes.

    Bending (10^(a1-a2*sigma)) and creep (creep polynomial) are
    stress-driven; the empirical operational life is flat. Converted to
    replacements per year using the model's annual cycle and flight rates.
    The governing (max-frequency) envelope is highlighted, with a marker
    at the V3's operating stress.
    """
    costs: TetherCosts = model.costs.tether
    flightHours, cycleCount = annual_rates(model)

    sigmaMaxGpa = costs.maxStress / PA_PER_GPA
    sigma = np.linspace(0.02, sigmaMaxGpa, 250)          # [GPa]

    fig, ax = plt.subplots(figsize=(9, 6))

    freqBend = freqCreep = None
    if costs.bendingLifeA1 is not None:
        nFail = 10.0 ** (costs.bendingLifeA1 - costs.bendingLifeA2 * sigma)
        freqBend = (costs.nBends * cycleCount) / nFail / BEND_LIFE_CORRECTION
        ax.plot(sigma, freqBend, label='Bending fatigue', color='#4C72B0')
    if costs.creepLifeCoefficients is not None:
        # The creep model treats 10^poly(sigma) as a life in years, so the
        # frequency is its reciprocal (no cycle-rate multiplication).
        creepLifeYears = 10.0 ** np.polyval(costs.creepLifeCoefficients, sigma)
        freqCreep = 1.0 / creepLifeYears
        ax.plot(sigma, freqCreep, label='Creep rupture', color='#55A868')

    envelope = None
    if costs.operationalLife is not None:
        freqOper = np.full_like(sigma, flightHours / costs.operationalLife)
        ax.plot(sigma, freqOper, label='Operational wear (empirical)',
                color='#C44E52', ls='--')
        modes = [freqOper] + [f for f in (freqBend, freqCreep)
                              if f is not None]
        envelope = np.max(np.vstack(modes), axis=0)
        ax.plot(sigma, envelope, color='black', lw=2.5, alpha=0.4,
                label='Governing (max) envelope')

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
    ax.set_xlabel('Fibre stress [GPa]')
    ax.set_ylabel('Replacement frequency [1/year]')
    ax.set_title('Tether life vs stress: fatigue governs high stress, '
                 'operational wear low stress')
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
    """Average and peak power vs wind speed with the wind PDF overlaid."""
    perf = model.inputs.performance
    metrics = eco['metrics']
    wind = np.asarray(perf.windSpeeds)
    avgKw = np.asarray(perf.averagePower) / 1e3

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(wind, avgKw, 'o-', color='#4C72B0', label='Average cycle power')
    ax.axhline(perf.ratedPower / 1e3, color='#C44E52', ls='--',
               label=f'Rated power ({perf.ratedPower / 1e3:.1f} kW)')
    if perf.peakMechanicalPower:
        ax.axhline(perf.peakMechanicalPower / 1e3, color='#8172B3', ls=':',
                   label=f'Peak reel-out ({perf.peakMechanicalPower / 1e3:.1f}'
                         f' kW)')
    ax.set_xlabel('Wind speed [m/s]')
    ax.set_ylabel('Power [kW]')
    ax.grid(True, alpha=0.3)

    axPdf = ax.twinx()
    axPdf.fill_between(wind, np.asarray(perf.windPdf), color='gray',
                       alpha=0.2, label='Wind PDF')
    axPdf.set_ylabel('Wind probability density [-]')
    axPdf.set_ylim(bottom=0)

    ax.set_title(f'Power curve and wind resource  '
                 f'(CF = {metrics["CF"]:.2f}, AEP = {metrics["AEP"]:.1f} '
                 f'MWh/yr)')
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = axPdf.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, fontsize=8, loc='upper left')
    fig.tight_layout()
    return _save(fig, outdir, '6_power_curve_wind')

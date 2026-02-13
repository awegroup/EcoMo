"""Display economic results.

This module provides functions to display economic metrics and breakdown
from the ECOMo simulation, including pie charts and cashflow plots
matching the MATLAB eco_displayResults.m output.
"""

import numpy as np
from typing import Dict, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def eco_display_results(inp: Dict[str, Any], eco: Dict[str, Any],
                        show: bool = True, save_dir: str = None) -> None:
    """Display economic metrics and breakdown.

    Generates two figures matching the MATLAB eco_displayResults output:
    - Figure 1: Three pie charts (ICC, OMC, LCoE)
    - Figure 2: Metrics text and cashflow bar chart

    Also prints compact output to the console.

    Args:
        inp: Dictionary containing input parameters.
        eco: Dictionary containing economic results and metrics.
        show: Whether to display plots interactively. Defaults to True.
        save_dir: If provided, save figures to this directory as PNG files.
    """
    m = eco['metrics']

    # ------------------------------------------------------------------ #
    #  Extract significant contributions (matching MATLAB thresholds)
    # ------------------------------------------------------------------ #
    icc_arr = np.array(m['icc'], dtype=float)
    omc_arr = np.array(m['omc'], dtype=float)
    lcoe_arr = np.array(m['LCoE_contr'], dtype=float)

    # ICC - components > 2 %
    icc_perc = icc_arr / icc_arr.sum() * 100
    icc_order = np.argsort(icc_perc)[::-1]
    sig_icc = icc_perc[icc_order] > 2
    list_icc = [m['icc_name'][i] + '.capex' for i in icc_order[sig_icc]]
    pp_icc = icc_perc[icc_order][sig_icc]

    # OMC - components > 2 %
    omc_perc = omc_arr / omc_arr.sum() * 100 if omc_arr.sum() > 0 else omc_arr * 0
    omc_order = np.argsort(omc_perc)[::-1]
    sig_omc = omc_perc[omc_order] > 2
    list_omc = [m['omc_name'][i] + '.opex' for i in omc_order[sig_omc]]
    pp_omc = omc_perc[omc_order][sig_omc]

    # LCoE - components > 3 %
    lcoe_perc = lcoe_arr / lcoe_arr.sum() * 100 if lcoe_arr.sum() > 0 else lcoe_arr * 0
    lcoe_order = np.argsort(lcoe_perc)[::-1]
    sig_lcoe = lcoe_perc[lcoe_order] > 3
    list_lcoe = [m['LCoE_contr_name'][i] for i in lcoe_order[sig_lcoe]]
    pp_lcoe = lcoe_perc[lcoe_order][sig_lcoe]

    # ------------------------------------------------------------------ #
    #  Plotting (requires matplotlib)
    # ------------------------------------------------------------------ #
    if HAS_MATPLOTLIB:
        fig1 = _plot_pie_charts(m, list_icc, pp_icc, list_omc, pp_omc,
                                list_lcoe, pp_lcoe)
        fig2 = _plot_metrics_cashflow(inp, m)
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            fig1.savefig(os.path.join(save_dir, 'cost_shares.png'),
                         dpi=150, bbox_inches='tight')
            fig2.savefig(os.path.join(save_dir, 'metrics_cashflow.png'),
                         dpi=150, bbox_inches='tight')
            print(f"Figures saved to {save_dir}/")
        if show:
            plt.show()
        else:
            plt.close('all')
    else:
        print("matplotlib not installed — skipping charts.")

    # ------------------------------------------------------------------ #
    #  Console output (matches MATLAB disp format)
    # ------------------------------------------------------------------ #
    print(f"ICC = {round(m['ICC'] / 1e3)} k\u20ac")
    print(f"OMC = {round(m['OMC'] / 1e3)} k\u20ac/year")
    print(f"LCoE = {round(m['LCoE'])} \u20ac/MWh")
    print(f"CoVE = {round(m['CoVE'])} \u20ac/MWh")
    print(f"LRoE = {round(m['LRoE'])} \u20ac/MWh")
    print(f"LPoE = {round(m['LPoE'])} \u20ac/MWh")
    print(f"NPV = {round(m['NPV'] / 1e3)} k\u20ac")
    print(f"IRR = {round(m['IRR'], 3) * 100} %")


# ====================================================================== #
#  Private helper functions
# ====================================================================== #

def _build_color_map(list_icc, list_omc, list_lcoe):
    """Build a consistent colour mapping across all three charts.

    Args:
        list_icc: ICC component labels.
        list_omc: OMC component labels.
        list_lcoe: LCoE component labels.

    Returns:
        Dictionary mapping component name to RGBA colour tuple.
    """
    all_components = list(dict.fromkeys(list_icc + list_omc + list_lcoe))
    n = max(len(all_components), 1)
    jetColors = cm.jet(np.linspace(0, 1, n))
    pastelFactor = 0.3
    pastelColors = (1 - pastelFactor) * jetColors[:, :3] + pastelFactor
    pastelColors = np.clip(pastelColors, 0, 1)
    return {name: pastelColors[i] for i, name in enumerate(all_components)}


def _plot_pie_charts(m, list_icc, pp_icc, list_omc, pp_omc,
                     list_lcoe, pp_lcoe):
    """Create Figure 1 with three pie charts side by side.

    Args:
        m: Metrics dictionary.
        list_icc: Significant ICC component labels.
        pp_icc: Corresponding ICC percentages.
        list_omc: Significant OMC component labels.
        pp_omc: Corresponding OMC percentages.
        list_lcoe: Significant LCoE component labels.
        pp_lcoe: Corresponding LCoE percentages.
    """
    colorMap = _build_color_map(list_icc, list_omc, list_lcoe)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Cost shares above 2%', fontsize=14)

    # ICC pie
    _single_pie(axes[0], pp_icc, list_icc, colorMap,
                f"CapEx = {round(m['ICC'] / 1e3)} k\u20ac")

    # OMC pie
    _single_pie(axes[1], pp_omc, list_omc, colorMap,
                f"OpEx = {round(m['OMC'] / 1e3)} k\u20ac/year")

    # LCoE pie
    _single_pie(axes[2], pp_lcoe, list_lcoe, colorMap,
                f"LCoE = {round(m['LCoE'])} \u20ac/MWh")

    fig.set_facecolor('w')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _single_pie(ax, percentages, labels, colorMap, title):
    """Draw one pie chart on the given axes.

    Args:
        ax: Matplotlib axes.
        percentages: Array of percentage values.
        labels: List of component labels.
        colorMap: Dictionary mapping label to colour.
        title: Title string for the chart.
    """
    if len(percentages) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(title, fontsize=12)
        return

    colors = [colorMap.get(l, (0.7, 0.7, 0.7)) for l in labels]
    ax.pie(percentages, autopct='%1.0f%%', colors=colors, startangle=90)
    ax.legend(labels, loc='lower center', fontsize=8,
              bbox_to_anchor=(0.5, -0.25))
    ax.set_title(title, fontsize=12)


def _plot_metrics_cashflow(inp, m):
    """Create Figure 2 with metrics text and cashflow bar chart.

    Args:
        inp: Input parameters dictionary.
        m: Metrics dictionary.
    """
    fig, (axText, axBar) = plt.subplots(2, 1, figsize=(8, 8))

    # Metrics text
    metricsStr = (
        f"Metrics:\n\n"
        f"CF = {m['CF']:.2f}\n"
        f"LCoE = {m['LCoE']:.0f} EUR/MWh\n"
        f"CoVE = {m['CoVE']:.0f} EUR/MWh\n"
        f"LRoE = {m['LRoE']:.0f} EUR/MWh\n"
        f"LPoE = {m['LPoE']:.0f} EUR/MWh\n"
        f"NPV = {m['NPV'] / 1e3:.0f} k EUR\n"
        f"IRR = {m['IRR']:.3f}"
    )
    axText.text(0.1, 0.9, metricsStr, fontsize=12, verticalalignment='top',
                family='monospace')
    axText.axis('off')

    # Cashflow bar chart
    cashflow = np.array(m['cashflow'])
    years = np.arange(len(cashflow))
    axBar.bar(years, cashflow / 1e6, color='steelblue', label='Cashflow')
    if m['payback_year'] is not None and m['payback_year'] < len(cashflow):
        py = m['payback_year']
        axBar.plot(py, cashflow[py] / 1e6, 'ro', markersize=8,
                   label='Payback Year')
    axBar.set_ylabel('Million \u20ac', fontsize=12)
    axBar.set_xlabel('Year', fontsize=12)
    axBar.set_xticks(years)
    axBar.set_xticklabels([str(y) for y in range(len(cashflow))])
    axBar.set_title('Project Cashflow', fontsize=12)
    axBar.legend(fontsize=10)
    axBar.grid(True)

    fig.set_facecolor('w')
    fig.tight_layout()
    return fig

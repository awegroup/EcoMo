"""Display economic results.

This module provides functions to display economic metrics and the
cost breakdown from the ECOMo simulation, including pie charts and
cashflow plots matching the MATLAB eco_displayResults.m output.
"""

import os
from typing import Dict, Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Components below these shares are hidden from the pie charts
ICC_SHARE_THRESHOLD = 2   # [%]
OMC_SHARE_THRESHOLD = 2   # [%]
LCOE_SHARE_THRESHOLD = 3  # [%]


def eco_display_results(eco: Dict[str, Any],
                        show: bool = True, save_dir: str = None) -> None:
    """Display economic metrics and breakdown.

    Generates two figures matching the MATLAB eco_displayResults
    output:

    - Figure 1: Three pie charts (ICC, OMC, LCoE)
    - Figure 2: Metrics text and cashflow bar chart

    Also prints compact output to the console.

    Args:
        eco (dict): Economic results and metrics.
        show (bool): Whether to display plots interactively. Defaults
            to True.
        save_dir (str): If provided, save figures to this directory as
            PNG files. Defaults to None.
    """
    metrics = eco['metrics']

    # ------------------------------------------------------------------ #
    #  Extract significant contributions (matching MATLAB thresholds)
    # ------------------------------------------------------------------ #
    iccArr = np.array(metrics['icc'], dtype=float)
    omcArr = np.array(metrics['omc'], dtype=float)
    lcoeArr = np.array(metrics['LCoE_contr'], dtype=float)

    # ICC - components > 2 %
    iccPerc = iccArr / iccArr.sum() * 100
    iccOrder = np.argsort(iccPerc)[::-1]
    sigIcc = iccPerc[iccOrder] > ICC_SHARE_THRESHOLD
    listIcc = [metrics['icc_name'][i] + '.capex' for i in iccOrder[sigIcc]]
    ppIcc = iccPerc[iccOrder][sigIcc]

    # OMC - components > 2 %
    omcPerc = (omcArr / omcArr.sum() * 100 if omcArr.sum() > 0
               else omcArr * 0)
    omcOrder = np.argsort(omcPerc)[::-1]
    sigOmc = omcPerc[omcOrder] > OMC_SHARE_THRESHOLD
    listOmc = [metrics['omc_name'][i] + '.opex' for i in omcOrder[sigOmc]]
    ppOmc = omcPerc[omcOrder][sigOmc]

    # LCoE - components > 3 %
    lcoePerc = (lcoeArr / lcoeArr.sum() * 100 if lcoeArr.sum() > 0
                else lcoeArr * 0)
    lcoeOrder = np.argsort(lcoePerc)[::-1]
    sigLcoe = lcoePerc[lcoeOrder] > LCOE_SHARE_THRESHOLD
    listLcoe = [metrics['LCoE_contr_name'][i] for i in lcoeOrder[sigLcoe]]
    ppLcoe = lcoePerc[lcoeOrder][sigLcoe]

    # ------------------------------------------------------------------ #
    #  Plotting (requires matplotlib)
    # ------------------------------------------------------------------ #
    if HAS_MATPLOTLIB:
        fig1 = _plot_pie_charts(metrics, listIcc, ppIcc, listOmc, ppOmc,
                                listLcoe, ppLcoe)
        fig2 = _plot_metrics_cashflow(metrics)
        if save_dir:
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
    print(f"ICC = {round(metrics['ICC'] / 1e3)} k€")
    print(f"OMC = {round(metrics['OMC'] / 1e3)} k€/year")
    print(f"LCoE = {round(metrics['LCoE'])} €/MWh")
    print(f"CoVE = {round(metrics['CoVE'])} €/MWh")
    print(f"LRoE = {round(metrics['LRoE'])} €/MWh")
    print(f"LPoE = {round(metrics['LPoE'])} €/MWh")
    print(f"NPV = {round(metrics['NPV'] / 1e3)} k€")
    print(f"IRR = {round(metrics['IRR'], 3) * 100} %")


# ====================================================================== #
#  Private helper functions
# ====================================================================== #

def _build_color_map(listIcc, listOmc, listLcoe):
    """Build a consistent colour mapping across all three charts.

    Args:
        listIcc (list): ICC component labels.
        listOmc (list): OMC component labels.
        listLcoe (list): LCoE component labels.

    Returns:
        dict: Mapping from component name to RGBA colour tuple.
    """
    allComponents = list(dict.fromkeys(listIcc + listOmc + listLcoe))
    n = max(len(allComponents), 1)
    jetColors = cm.jet(np.linspace(0, 1, n))
    pastelFactor = 0.3
    pastelColors = (1 - pastelFactor) * jetColors[:, :3] + pastelFactor
    pastelColors = np.clip(pastelColors, 0, 1)
    return {name: pastelColors[i] for i, name in enumerate(allComponents)}


def _plot_pie_charts(metrics, listIcc, ppIcc, listOmc, ppOmc,
                     listLcoe, ppLcoe):
    """Create Figure 1 with three pie charts side by side.

    Args:
        metrics (dict): Metrics dictionary.
        listIcc (list): Significant ICC component labels.
        ppIcc (np.ndarray): Corresponding ICC percentages.
        listOmc (list): Significant OMC component labels.
        ppOmc (np.ndarray): Corresponding OMC percentages.
        listLcoe (list): Significant LCoE component labels.
        ppLcoe (np.ndarray): Corresponding LCoE percentages.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    colorMap = _build_color_map(listIcc, listOmc, listLcoe)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Cost shares above 2%', fontsize=14)

    # ICC pie
    _single_pie(axes[0], ppIcc, listIcc, colorMap,
                f"CapEx = {round(metrics['ICC'] / 1e3)} k€")

    # OMC pie
    _single_pie(axes[1], ppOmc, listOmc, colorMap,
                f"OpEx = {round(metrics['OMC'] / 1e3)} k€/year")

    # LCoE pie
    _single_pie(axes[2], ppLcoe, listLcoe, colorMap,
                f"LCoE = {round(metrics['LCoE'])} €/MWh")

    fig.set_facecolor('w')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _single_pie(ax, percentages, labels, colorMap, title):
    """Draw one pie chart on the given axes.

    Args:
        ax: Matplotlib axes.
        percentages (np.ndarray): Percentage values.
        labels (list): Component labels.
        colorMap (dict): Mapping from label to colour.
        title (str): Title string for the chart.
    """
    if len(percentages) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(title, fontsize=12)
        return

    colors = [colorMap.get(label, (0.7, 0.7, 0.7)) for label in labels]
    ax.pie(percentages, autopct='%1.0f%%', colors=colors, startangle=90)
    ax.legend(labels, loc='lower center', fontsize=8,
              bbox_to_anchor=(0.5, -0.25))
    ax.set_title(title, fontsize=12)


def _plot_metrics_cashflow(metrics):
    """Create Figure 2 with metrics text and cashflow bar chart.

    Args:
        metrics (dict): Metrics dictionary.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    fig, (axText, axBar) = plt.subplots(2, 1, figsize=(8, 8))

    # Metrics text
    metricsStr = (
        f"Metrics:\n\n"
        f"CF = {metrics['CF']:.2f}\n"
        f"LCoE = {metrics['LCoE']:.0f} EUR/MWh\n"
        f"CoVE = {metrics['CoVE']:.0f} EUR/MWh\n"
        f"LRoE = {metrics['LRoE']:.0f} EUR/MWh\n"
        f"LPoE = {metrics['LPoE']:.0f} EUR/MWh\n"
        f"NPV = {metrics['NPV'] / 1e3:.0f} k EUR\n"
        f"IRR = {metrics['IRR']:.3f}"
    )
    axText.text(0.1, 0.9, metricsStr, fontsize=12, verticalalignment='top',
                family='monospace')
    axText.axis('off')

    # Cashflow bar chart
    cashflow = np.array(metrics['cashflow'])
    years = np.arange(len(cashflow))
    axBar.bar(years, cashflow / 1e6, color='steelblue', label='Cashflow')
    if (metrics['payback_year'] is not None and
            metrics['payback_year'] < len(cashflow)):
        paybackYear = metrics['payback_year']
        axBar.plot(paybackYear, cashflow[paybackYear] / 1e6, 'ro',
                   markersize=8, label='Payback Year')
    axBar.set_ylabel('Million €', fontsize=12)
    axBar.set_xlabel('Year', fontsize=12)
    axBar.set_xticks(years)
    axBar.set_xticklabels([str(y) for y in range(len(cashflow))])
    axBar.set_title('Project Cashflow', fontsize=12)
    axBar.legend(fontsize=10)
    axBar.grid(True)

    fig.set_facecolor('w')
    fig.tight_layout()
    return fig

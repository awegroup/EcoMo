# Scripts

Entry points and helper scripts for ECOMo, grouped by audience.

## Runners (general)

Run the model on a configuration. Useful for any user.

- `run_ecomo.py` — run the model on the example V3.25 case
  (`config/example/`) and write the results.
- `run_ecomo_from_awespa.py` — run the model on AWESPA power-model outputs.

## `thesis_figures/` (V3.25 case-specific)

Figures for the TU Delft V3.25 study. They hard-code the example paths and
produce thesis plots; use them as templates rather than general tools.

- `fig_maturity_roadmap_V3.py` — LCoE vs development stage (early/mid/mature).
- `fig_lcoe_improvement_roadmap.py` — LCoE vs a continuous improvement factor.
- `fig_replacement_shares.py` — per-component replacement-cost composition.

## `tools/` (one-off)

- `convert_excel_to_yaml.py` — one-time conversion of the original Excel
  input files to the YAML config format. Requires `pip install .[convert]`.

# ECOMo — Economic Model for Airborne Wind Energy Systems

Python implementation of the AWE-Eco reference economic model, originally developed in MATLAB by the AWE Group at TU Delft.

ECOMo computes capital expenditure (CAPEX), operational expenditure (OPEX), levelised cost of energy (LCoE), net present value (NPV), internal rate of return (IRR), and other economic metrics for airborne wind energy systems.

## :gear: Installation

### Dependencies

- Python >= 3.10
- numpy, scipy, matplotlib, pyyaml
- pandas, openpyxl (only for re-running the one-off Excel-to-YAML
  converter: `pip install .[convert]`)

### Installation Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/awegroup/ecomo.git
    cd ecomo
    ```

2. Create and activate a virtual environment:

   Linux or Mac:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    Windows:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3. Install the package:

   For users:
    ```bash
    pip install .
    ```

   For developers:
    ```bash
    pip install -e .[dev]
    ```

## :eyes: Usage

The model is configured from an awesIO YAML settings file and run through
the `EcoMo` class. All inputs are YAML files; the settings
file references three input files per topology case:

- a **system** file (awesIO `system_schema.yml` format) with the physical
  parameters (wing mass and area, tether geometry, storage capacities),
- a **cost inputs** file with all cost model parameters and market data,
- a **performance** file with the wind-dependent performance-model
  outputs (power curve, cycle time, tether force).

### Standalone mode

AEP and system performance data come from the performance YAML referenced
by the settings file:

```bash
python scripts/run_ecomo.py
```

```python
from ecomo import EcoMo

model = EcoMo()
model.load_configuration(
    economic_settings_path="config/example/economic_settings_GG_fixed.yml",
)
results = model.compute_economics(
    output_path="results/example/ecomo_results.yml",
    verbose=True,
)
```

To switch between system configurations (GG/FG, fixed/soft wing), point
`load_configuration()` to one of the settings files in `config/example/`:

- `economic_settings_GG_fixed.yml` — ground-gen, fixed wing
- `economic_settings_GG_soft.yml` — ground-gen, soft wing
- `economic_settings_FG.yml` — fly-gen

To analyse a different system, swap the file references in the settings
`input_files` section (paths are resolved relative to the settings file).

### AWESPA-connected mode

AEP and power curve data are taken from AWESPA output files
(`aep_results.yml` and `power_curves.yml`) instead of the performance
file. This mode is selected entirely from the settings file: set
`input_files.aep_results` and `input_files.power_curves` (both, or
neither). The example `economic_settings_GG_fixed_awespa.yml` does this:

```bash
python scripts/run_ecomo_from_awespa.py
```

```python
model = EcoMo()
model.load_configuration(
    economic_settings_path="config/example/economic_settings_GG_fixed_awespa.yml",
)
results = model.compute_economics(
    output_path="results/example/ecomo_results_awespa.yml",
    verbose=True,
)
```

Note: the tether force is not available from AWESPA power model outputs.
In AWESPA-connected mode it must be specified in the economic settings
YAML (`system_extras.tether_force_override`, as a list or constant),
otherwise tether force-based replacement costs are omitted with a
warning. The same applies to the peak mechanical power and the storage
exchanged energy, which fall back to the approximations from the
reference report (2.5 x rated power, half the rated storage capacity).

### Wind resource

The economic integrals (electricity price, replacement frequencies, and
AEP in standalone mode) are weighted by the wind-speed distribution. By
default this is a Weibull distribution set by `wind_resource.weibull_shape`
and `wind_resource.weibull_scale`. To use the actual site distribution,
set `wind_resource.resource_file` to an awesIO wind resource file
(`wind_resource_schema.yml`); its marginal wind-speed distribution then
drives the integrals. The AWESPA example uses the same site file AWESPA
used (`data/power_law_case_1.yml`), so the economic weighting matches the
power model. AEP in AWESPA-connected mode is taken directly from
`aep_results.yml` and is unaffected by this choice.

### Project Structure

```
ecomo/
├── config/
│   └── example/          # YAML inputs, per topology case:
│       ├── economic_settings_*.yml      # settings (references the files below)
│       ├── economic_cost_inputs_*.yml   # cost model parameters + market data
│       ├── system_*.yml                 # physical system (awesIO system_schema)
│       └── system_performance_*.yml     # performance-model outputs
├── data/                 # Legacy Excel inputs (converter source only)
│                         # + example AWESPA output files
├── docs/                 # Sphinx documentation
├── notebooks/            # Jupyter notebooks
├── results/              # Output files (generated at runtime)
├── scripts/
│   ├── run_ecomo.py             # Entry point (standalone mode)
│   ├── run_ecomo_from_awespa.py # Entry point (AWESPA-connected mode)
│   └── convert_excel_to_yaml.py # One-off Excel-to-YAML migration
├── src/
│   └── ecomo/            # Python package
│       ├── base.py              # EconomicModel abstract base class
│       ├── ecomo_economic.py    # EcoMo concrete class
│       └── ecomo/               # Subsystem cost modules
│           ├── eco_main.py             # Orchestrator
│           ├── eco_inputs.py           # Typed input dataclasses
│           ├── eco_costs.py            # Typed cost-parameter dataclasses
│           ├── eco_kite.py
│           ├── eco_tether.py
│           ├── eco_gstation.py
│           ├── eco_bos.py
│           ├── eco_bop.py
│           ├── eco_metrics.py
│           ├── eco_load_cost_inputs.py
│           ├── eco_load_system.py
│           ├── eco_load_performance.py
│           ├── eco_load_wind_resource.py
│           ├── eco_wind.py
│           ├── constants.py
│           └── eco_display_results.py
├── tests/
├── pyproject.toml
└── README.md
```

## :wave: Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## :warning: License and Waiver

MIT License

> Technische Universiteit Delft hereby disclaims all copyright interest in the program "ECOMo" (Economic Model for Airborne Wind Energy Systems) written by the Author(s).
>
> Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering
>
> Copyright (c) 2026 TU Delft.

## :book: References

This software is a Python implementation of the AWE-Eco model. If you use it in your work, please cite the original report:

> Rishikesh Joshi and Filippo Trevisi (2024) "Reference Economic Model for Airborne Wind Energy Systems" (Version 1). IEA Wind TCP Task 48. https://doi.org/10.5281/zenodo.10959930.

## Help and Documentation
[AWE Group | Developer Guide](https://awegroup.github.io/developer-guide/)

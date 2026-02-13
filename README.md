# ECOMo — Economic Model for Airborne Wind Energy Systems

Python implementation of the AWE-Eco reference economic model, originally developed in MATLAB by the AWE Group at TU Delft.

ECOMo computes capital expenditure (CAPEX), operational expenditure (OPEX), levelised cost of energy (LCoE), net present value (NPV), internal rate of return (IRR), and other economic metrics for airborne wind energy systems.

## :gear: Installation

### Dependencies

- Python >= 3.10
- numpy, pandas, openpyxl, scipy, matplotlib

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

Run the example script:

```bash
python src/run_ecomo.py
```

To switch between system configurations, edit the `configure()` call in
`src/run_ecomo.py`:

```python
from ecomo.config import configure

# Ground-gen, soft wing
configure(
    name='example_system',
    input_cost_file='eco_cost_inputs_GG_soft',
    input_model_file='eco_system_inputs_GG_soft_example',
    power='GG',
    wing='soft',
)

# Ground-gen, fixed wing
configure(
    name='example_system',
    input_cost_file='eco_cost_inputs_GG_fixed',
    input_model_file='eco_system_inputs_GG_fixed_example',
    power='GG',
    wing='fixed',
)
```

### Project Structure

```
ecomo/
├── data/                 # Excel input files (cost & system parameters)
├── docs/                 # Sphinx documentation
├── notebooks/            # Jupyter notebooks
├── results/              # Output files (generated at runtime)
├── src/
│   ├── run_ecomo.py      # Main entry point script
│   └── ecomo/            # Python package
│       ├── config.py
│       ├── eco_system_inputs.py
│       ├── eco_import_model.py
│       ├── eco_import_cost_par.py
│       ├── eco_main.py
│       ├── eco_kite.py
│       ├── eco_tether.py
│       ├── eco_gstation.py
│       ├── eco_bos.py
│       ├── eco_bop.py
│       ├── eco_metrics.py
│       └── eco_display_results.py
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

## :gem: Help and Documentation
[AWE Group | Developer Guide](https://awegroup.github.io/developer-guide/)

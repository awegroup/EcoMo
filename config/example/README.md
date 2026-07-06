# Example configuration

Production-ready example for the AWESPA-connected workflow: the **TU Delft
V3.25** soft kite evaluated against the V3 AWESPA power model outputs.

Files:

- `tudelft V3_25.yml` — the V3.25 system (awesIO `system_schema.yml`).
- `economic_cost_inputs_V3.yml` — soft-wing cost model parameters.
- `economic_settings_V3_example.yml` — the run settings; points at the V3
  AWESPA outputs in `../../data/` (`aep_results.yml`, `power_curves.yml`) and
  the site wind resource (`power_law_case_1.yml`), which activates
  AWESPA-connected mode (the `performance` file is ignored).

Run it with:

```
python scripts/run_ecomo.py
```

Results are written to `results_example/ecomo_results.yml`.

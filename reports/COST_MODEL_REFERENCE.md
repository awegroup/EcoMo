# ECOMo cost model reference

Background and modelling rationale for the ECOMo economic model. The source
code is kept self-explanatory (Google-style docstrings, no inline
derivations), so this document holds the equations, assumptions and
citations that would otherwise clutter the code.

ECOMo is a bottom-up conceptual-design cost model for airborne wind energy
(AWE) systems. Each subsystem cost is derived from the physical design and
annualised into a levelised cost of energy (LCoE). It targets a mature,
well-operated system by default, and can also represent an early-maturity
prototype through the development-stage inputs (see below).

## Levelised cost of energy

```
LCoE = (CRF * ICC + OMC) / AEP
```

- `ICC` — initial capital cost, the sum of every `CAPEX` leaf.
- `OMC` — annual operating cost, the sum of every `OPEX` leaf.
- `CRF` — capital recovery factor from the weighted average cost of capital
  (WACC), derived from the debt/equity split, the costs of debt and equity
  and the tax rate.
- `AEP` — annual energy production, from the AWESPA power model and the wind
  resource, scaled by availability.

Only the `CAPEX` term is discounted through the CRF; `OPEX` is charged
directly each year. Replacement of a wearing part enters as an `OPEX` equal
to its replacement frequency times its `CAPEX`.

Cost aggregation sums only the leaves named exactly `CAPEX` and `OPEX`; every
other value in the results tree is a diagnostic and is ignored by the
totals.

## Kite

Two-term soft-wing structure cost:

```
C_kite = C_mat_ref * (S / S_ref)^b_mat + C_lab * S
```

with wing area `S`, a reference-area material cost `C_mat_ref` at `S_ref`, a
material scaling exponent `b_mat` and a per-area labour coefficient `C_lab`.

The airborne electronics are a scaled KCU/avionics term plus a separate
sensor line. Canopy replacement uses one of two models:

- **Reel-out-hour model (preferred):** the canopy life is set in loaded
  (reel-out) flight hours; the replacement frequency is the annual reel-out
  hours divided by that life. An optional load exponent weights the hours by
  `(F / F_ref)^m` (Miner's rule).
- **Calendar model (legacy):** the structural life is set in flying years.

## Tether

CAPEX is mass-based:

```
mass  = area * fibre_area_fraction * length * density * (1 + coating_fraction)
CAPEX = price_mass * mass          (GG)
CAPEX = manufacturing_factor * price_mass * mass   (FG, conductive)
```

The tether replacement frequency is the highest of three life modes (the
shortest life governs):

### Bending fatigue — HMPE bearing-pressure master curve

The preferred bending model is the Meuwissen/Bosman (2024) HMPE
bearing-pressure master curve, calibrated to 6 mm SK75 data:

```
p_N  = k_pw * sigma[MPa] / (D/d)          bearing pressure, clamped to pw_limit
N_f  = C * p_N^(-B)                        cycles to failure
```

with fibre stress `sigma`, the winch drum-to-tether diameter ratio `D/d`, a
bearing-pressure coefficient `k_pw`, a low-pressure clamp `pw_limit`, and the
fitted coefficients `C` and `B`. The cycles-to-failure feed the usual
wind-distribution damage integral to give the bending life. A design safety
factor `SF` retires the tether at `N_f / SF`; it scales the bending
replacement frequency and applies **only** while the master curve is active.

Calibration (V3.25): `C = 5e6`, `B = 2.0`, `k_pw = 1.0`, `pw_limit = 8 MPa`,
`SF = 3.0`, `D/d = 30`. This gives a V3 bending life of ~1300 flight-h before
SF and ~430 after.

`D/d` comes from `ground_station.winch.drum_to_tether_diameter_ratio`; the
bending life therefore responds to the winch drum sizing, which is what makes
it usable in the TEF design optimisation where the tether diameter is free.

### Bending fatigue — legacy a1/a2 model (fallback)

When the master curve is not configured, a semi-log S-N model is used:

```
log10(N_f) = a1 - a2 * sigma[GPa]
```

`a1` is interpolated log-linearly in `D/d` from a small calibration table.
This fallback is diameter-independent in the regime where the maturity
roadmap uses it, so the design safety factor does **not** derate it.

### Creep

A polynomial in `sigma[GPa]` gives the creep-rupture life, integrated over
the wind distribution.

### Operational wear (opt-in)

Non-fatigue degradation (UV, abrasion, particle ingress, handling) is modelled
as an empirical life in flight hours, consumed at the annual flight-hour rate.
It is opt-in via `operational_life_enabled`. When enabled it can govern at the
low stress of a soft-wing system, decoupling the tether life from the bending
mode; this is the mode used by the maturity roadmap (master curve off). When
disabled, bending/creep fatigue governs; this is the mode used by the design
optimisation (master curve on).

## Ground station

Bottom-up component costs: winch (drum sized from the tether at `D/d`),
gearbox, generator, electrical storage (ultracapacitor or battery, sized from
the per-cycle exchanged energy), power converter, and the launch-and-land
system (priced per unit ground area). Components with a finite service life
charge a replacement `OPEX` for the re-buys within the project lifetime.

## Balance of system (BoS)

Three separate annual operating terms, each its own traceable leaf:

- **`BoS.OM`** — the per-kW O&M overhead
  (`operations_maintenance.price_power * rated_power_kW`), the
  size-dependent standing upkeep. Not labour.
- **`BoS.labour`** — the operator and maintenance crew:
  - operation (base crew), per operating day and size-independent:
    `C_op = (1 - automation) * w * N_op * h_op`, with the labour rate `w`,
    the annual operating days `N_op = f_wind * 365` (the windy fraction of
    the year), and the operating hours per day `h_op`;
  - maintenance, per flight hour:
    `C_maint = w * h_maint_per_flight_h * H_flight`, accruing with the annual
    flight hours `H_flight` rather than with the calendar.

The one-off BoS CAPEX (site preparation, foundation, installation,
decommissioning) is annualised through the CRF. The foundation is sized by the
peak mechanical power, not the rated electrical power.

## Availability

Availability scales the annual flight hours (canopy/tether wear and the
per-flight-hour maintenance) and, through the metrics, the net AEP — so
downtime raises cost and lowers energy consistently. It does not change the
operating-day count, since the crew is a standing cost tied to the operating
calendar.

## Development stages

A single `development_stage` setting (`early`, `mid`, `mature`) selects a
bundle of maturity-linked inputs. Any value set explicitly in the config
overrides the preset. The tether replacement model (bending master curve vs.
staged operational life) is independent of the stage; the preset only supplies
the operational-life value.

| stage  | canopy life (h) | operating h/day | maintenance h/flight-h | tether operational life (h) |
|--------|-----------------|-----------------|------------------------|-----------------------------|
| early  | 100             | 3.27            | 0.25                   | 250                         |
| mid    | 500             | 0.82            | 0.10                   | 1000                        |
| mature | 5000            | 0.16            | 0.01                   | 5000                        |

The operating hours per day encode weekly targets (20 / 5 / 1 h per week) at
the V3 operating-day count `N_op ~ 318`.


## Two use cases

- **Design optimisation (TEF):** the tether diameter is a free variable, so
  the master curve is on and the operational-wear mode is off — the
  diameter-dependent bending life governs. Runs at mid maturity.
- **Maturity roadmap:** the design is fixed and the master curve is off, so
  the flat staged operational life governs the tether. The stages step
  through the table above to show how LCoE falls as the technology matures.

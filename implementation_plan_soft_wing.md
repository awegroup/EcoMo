# ECOMo Soft-Wing Adaptation — Implementation Plan

## Context

This codebase (`ecomo`) is a techno-economic model for airborne wind energy systems.
It currently supports fixed-wing and soft-wing GG/FG topologies but several parts of
the soft-wing economic model still use fixed-wing assumptions or placeholder values.
This document specifies all changes needed to fully adapt the economic model for
soft-wing LEI kite assessment and later integration into the MDAO framework.

The performance data (power curves, AEP) comes from AWESPA / InertiaFree QSM and is
read from `data/power_curves.yml` and `data/aep_results.yml`.
The system configuration is in `config/test/gg_soft/`.

---

## Change 1 — Kite cost model (two-term formula)

**Why:** The current model uses a flat `(p_fabric + p_bridle) × S = €53/m²` with no
scaling exponent. Physical analysis shows material cost scales super-linearly with area
(structural components — struts, bridle, reinforcements — scale at b=1.5 under stress
similarity), while labour scales more nearly linearly. A two-term decomposition is more
accurate for trend spotting and is internally consistent with the thesis mass model.

**Formula:**

```
C_kite(S) = C_mat_ref × (S / S_ref)^b_mat  +  C_lab × S
```

where:
- `C_mat_ref` = material cost at reference area S_ref (EUR)
- `S_ref` = reference flat wing area (m²), default 60 m²
- `b_mat` = material scaling exponent, default 1.14 (cost-weighted from component model)
- `C_lab` = labour cost coefficient (EUR/m²), default 13.0
  (= seam density 10 m/m² × sewing cost €1.30/m)

Scaling exponent uncertainty band:
- Lower bound: b=1.00 (pure geometric area scaling)
- Reference: b=1.14 (component cost-weighted average, thesis Table 4.3)
- Upper bound: b=1.22 (mass model least-squares fit, thesis Eq. 4.42)

The anchor (`C_mat_ref` at `S_ref = 60 m²`) has two scenarios:
- Conservative (prototype as-built): C_mat_ref = €4,146 (from €4,926 total − €780 labour)
- Lean AWE-optimised: C_mat_ref = €3,220 (from €4,000 adjusted − €780 labour)
Use the lean anchor as default (Scenario B).

**Files to change:**

### `src/ecomo/ecomo/eco_costs.py`

Add the following fields to `KiteCosts`, replacing `priceFabric` and `priceBridle`:

```python
# Replace:
#   priceFabric: Optional[float] = None
#   priceBridle: Optional[float] = None
# With:
materialCostRef: Optional[float] = None        # C_mat at S_ref [EUR]
referenceArea: Optional[float] = None          # S_ref [m²]
materialScalingExponent: Optional[float] = None  # b_mat [-]
labourCostCoefficient: Optional[float] = None  # C_lab [EUR/m²]
```

Keep `priceFabric` and `priceBridle` as Optional with default None for backward
compatibility with fixed-wing configs that do not set them.

### `src/ecomo/ecomo/subsystems/eco_kite.py`

Replace the soft-wing CAPEX calculation:

```python
# OLD:
capex = (costs.priceFabric + costs.priceBridle) * kite.flatArea

# NEW:
if costs.materialCostRef is not None:
    # Two-term scaling model (soft-wing LEI)
    # TODO: C_mat_ref and S_ref are anchored to the 60m² LEI reference kite.
    #       Two anchor scenarios exist:
    #         Conservative (as-built prototype): C_mat_ref = 4146 EUR
    #         Lean AWE-optimised:                C_mat_ref = 3220 EUR
    #       Default uses the lean anchor (Scenario B from cost model derivation).
    #       Update if more representative cost data become available.
    # TODO: b_mat = 1.14 is the cost-weighted component average from thesis Table 4.3.
    #       Uncertainty band: b=1.00 (lower, pure area) to b=1.22 (upper, mass model).
    #       Sensitivity to this exponent is significant for large kite areas (>100 m²).
    # TODO: C_lab = 13.0 EUR/m² embeds sewing cost €1.30/m at 10 m/m² seam density.
    #       The €1.30/m rate is specific to the source data labour rate (~€7/h).
    #       For different manufacturing contexts, update via: C_lab = (p_labor/v_sew)*C_s
    #       where v_sew = 5.4 m/h (empirically constant across architectures and scales).
    s_ref = costs.referenceArea
    b = costs.materialScalingExponent
    capex = (costs.materialCostRef * (kite.flatArea / s_ref) ** b
             + costs.labourCostCoefficient * kite.flatArea)
else:
    # Fallback: original flat-price model (Joshi & Trevisi 2024)
    capex = (costs.priceFabric + costs.priceBridle) * kite.flatArea
```

### `src/ecomo/ecomo/loaders/eco_load_cost_inputs.py`

In `_load_kite_costs`, update the soft-wing section to load the new fields:

```python
if 'soft' in structure:
    soft = structure['soft']
    # New two-term model fields (preferred)
    if 'material_cost_ref' in soft:
        fields['materialCostRef'] = soft['material_cost_ref']
        fields['referenceArea'] = _require(soft, 'reference_area',
                                           'costs.kite.structure.soft')
        fields['materialScalingExponent'] = _require(
            soft, 'material_scaling_exponent', 'costs.kite.structure.soft')
        fields['labourCostCoefficient'] = _require(
            soft, 'labour_cost_coefficient', 'costs.kite.structure.soft')
    else:
        # Fallback to old flat-price model
        fields['priceFabric'] = _require(soft, 'price_fabric',
                                         'costs.kite.structure.soft')
        fields['priceBridle'] = _require(soft, 'price_bridle',
                                         'costs.kite.structure.soft')
    fields['structureLifetime'] = _require(soft, 'lifetime_flying_years',
                                           'costs.kite.structure.soft')
```

### `config/test/gg_soft/economic_cost_inputs_GG_soft.yml`

Replace the `soft` section:

```yaml
costs:
  kite:
    structure:
      soft:
        # Two-term kite cost model: C_kite = C_mat_ref*(S/S_ref)^b_mat + C_lab*S
        # Anchor: lean AWE-optimised 60m² LEI kite (Scenario B)
        # TODO: C_mat_ref = 3220 EUR is expert-estimated lean design cost minus labour.
        #       Conservative prototype anchor = 4146 EUR (use if no design optimisation).
        #       Update when production cost data for AWE-specific kites become available.
        material_cost_ref: 3220.0       # [EUR] material cost at reference area
        reference_area: 60.0            # [m²]  S_ref
        material_scaling_exponent: 1.14 # [-]   b_mat; range 1.00–1.22
        # TODO: labour_cost_coefficient = 13.0 EUR/m² assumes seam density 10 m/m²
        #       and sewing cost €1.30/m (= €7/h labour rate / 5.4 m/h sewing speed).
        #       Sewing speed 5.4 m/h is empirically constant across all kite architectures
        #       and scales. Seam density should be verified against V3 build records.
        labour_cost_coefficient: 13.0   # [EUR/m²] C_lab
        # TODO: lifetime_flying_years = 0.02853 corresponds to 250h (midpoint of
        #       observed 100–500h range). Target lifetime is 5000–10000h but not yet
        #       achieved. Update as empirical wear data from extended test campaigns
        #       become available. Current value means ~13 replacements/year at 3500 h/yr.
        lifetime_flying_years: 0.02853  # [yr] = 250h / 8760h
```

---

## Change 2 — Loading factor: time-weighted formula

**Why:** The current loading factor formula (from Joshi & Trevisi 2024, Eq. 4) is:

```
LF = ∫ f(v_w) × F_t(v_w) / F_t,max  dv_w
```

For a pumping kite this is a logical error: the kite only accumulates structural fatigue
during the reel-out (traction) phase, not during reel-in. The unweighted formula gives
LF = 0.355 vs. the correct time-weighted LF = 0.219 — a 38% overestimate in replacement
frequency that propagates directly into OpEx.

The corrected formula is:

```
LF = ∫ f(v_w) × (F_t(v_w) / F_t,max) × (t_reel_out(v_w) / t_cycle(v_w))  dv_w
```

All inputs are available from `power_curves.yml` (`reel_out_time` and `cycle_time`
per wind speed in the `timing` section of each wind speed entry).

**Files to change:**

### `src/ecomo/ecomo/eco_inputs.py`

Add `reelOutTimeFraction` to `PerformanceData`:

```python
@dataclass
class PerformanceData:
    ...
    reelOutTimeFraction: Optional[np.ndarray] = None
    # Array of t_reel_out / t_cycle per wind speed [-]; None for FG systems or when
    # timing data are unavailable (falls back to unweighted LF with a warning).
```

### `src/ecomo/ecomo_economic.py`

In `_parse_power_curves`, extract reel-out time fraction and peak reel-out power
alongside the existing fields:

```python
# Add to accumulation loop:
reelOutTimeSum = np.zeros(nSpeeds)
peakReelOutPower = 0.0

for curve in curves:
    ...
    for i, entry in enumerate(entries):
        if not entry.get('successful', False):
            continue
        performance = entry['performance']
        ...
        timing = performance['timing']
        cycle_t = timing.get('cycle_time', 0.0)
        reel_out_t = timing.get('reel_out_time', 0.0)
        if cycle_t > 0:
            reelOutTimeSum[i] += weight * reel_out_t / cycle_t
        # Peak reel-out power: take max over profiles and wind speeds
        reel_out_pwr = performance['power'].get('average_reel_out_power', 0.0)
        peakReelOutPower = max(peakReelOutPower, reel_out_pwr)

# Compute weighted average reel-out fraction
with np.errstate(divide='ignore', invalid='ignore'):
    reelOutFraction = np.where(timeWeight > 0, reelOutTimeSum / timeWeight, 0.0)

return {
    'windRange': windRange,
    'peAvg': peAvg,
    'dtCycle': dtCycle,
    'peRated': float(np.max(peAvg)),
    'reelOutFraction': reelOutFraction,    # NEW
    'peakReelOutPower': peakReelOutPower,  # NEW
}
```

In `_awespa_performance`, pass `reelOutTimeFraction` to `PerformanceData` and use
actual peak reel-out power instead of the 2.5× fallback:

```python
# Replace:
#   peakMechanicalPower = (peak_mechanical_power_fallback(data['peRated'])
#                          if topology.power == 'GG' else None)
# With:
if topology.power == 'GG':
    peakMechanicalPower = data.get('peakReelOutPower')
    if not peakMechanicalPower:
        peakMechanicalPower = peak_mechanical_power_fallback(data['peRated'])
else:
    peakMechanicalPower = None

performance = PerformanceData(
    ...
    peakMechanicalPower=peakMechanicalPower,
    reelOutTimeFraction=data.get('reelOutFraction'),  # NEW
    ...
)
```

### `src/ecomo/ecomo/subsystems/eco_kite.py`

Update `_soft_structure_replacement_frequency`:

```python
def _soft_structure_replacement_frequency(kite, costs, performance):
    if (performance.reelOutTimeFraction is not None
            and np.any(performance.reelOutTimeFraction > 0)):
        # Time-weighted loading factor for pumping kites (corrected formula).
        # Only the reel-out (traction) phase loads the kite structure.
        # Ref: identified as logical error in Joshi & Trevisi (2024) Eq.4 for
        # pumping kites; unweighted LF = 0.355 vs time-weighted LF = 0.219 (-38%).
        # TODO: reel-in phase loading is assumed negligible (loads ~10-20% of max).
        #       This is valid for typical soft-wing pumping kites; revisit if the
        #       retraction strategy changes significantly.
        loadFactor = np.trapezoid(
            performance.windPdf
            * performance.tetherForce / np.max(performance.tetherForce)
            * performance.reelOutTimeFraction,
            performance.windSpeeds)
    else:
        # Fallback: unweighted LF (Joshi & Trevisi 2024, Eq.4).
        # WARNING: This overestimates replacement frequency by ~38% for pumping kites
        # because it does not account for the reel-in phase having near-zero loading.
        # Provide reel_out_time in the power curves YAML to use the corrected formula.
        import warnings
        warnings.warn(
            "reel_out_time_fraction not available; using unweighted loading "
            "factor (Joshi & Trevisi 2024, Eq.4). This overestimates soft-kite "
            "replacement frequency by ~38% for pumping kites. Provide "
            "'reel_out_time' in power_curves.yml to use the corrected formula.",
            UserWarning, stacklevel=2,
        )
        loadFactor = np.trapezoid(
            performance.windPdf * performance.tetherForce /
            np.max(performance.tetherForce),
            performance.windSpeeds)

    return loadFactor / costs.structureLifetime
```

---

## Change 3 — KCU / Avionics: area-dependent scaling

**Why:** The current model uses a fixed €30,000 for avionics regardless of kite size.
Physical analysis (Braun 2015) shows KCU actuator requirements scale with tether force
≈ W_l × S. Motor cost scales approximately linearly with torque. A fixed + variable
split better captures this: electronics/sensors are fixed; mechanical actuators scale.

**Formula:**

```
C_avionics(S) = C_avionics_fixed  +  C_avionics_var_ref × (S / S_avionics_ref)^β
```

where:
- `C_avionics_fixed` = fixed electronics portion (~30% of total), default €2,400
- `C_avionics_var_ref` = variable actuator portion at S_ref (~70% of total), default €5,600
- `S_avionics_ref` = reference area, default 60 m² (same as kite cost model)
- `β` = scaling exponent, default 1.0 (linear with area)

### `src/ecomo/ecomo/eco_costs.py`

Add to `KiteCosts`:

```python
avionicsCostFixed: Optional[float] = None      # Fixed electronics [EUR]
avionicsCostVarRef: Optional[float] = None     # Variable actuators at S_ref [EUR]
avionicsReferenceArea: Optional[float] = None  # S_ref for avionics scaling [m²]
avionicsScalingExponent: Optional[float] = None  # β [-]
```

### `src/ecomo/ecomo/subsystems/eco_kite.py`

Update the avionics cost calculation:

```python
# Avionics
if costs.avionicsCostFixed is not None:
    # Scaled KCU model: fixed electronics + variable actuator portion
    # TODO: C_avionics_fixed = 2400 EUR (30% of 8000 EUR reference total).
    #       C_avionics_var_ref = 5600 EUR (70% of 8000 EUR reference total).
    #       Split ratio from Braun (2015) product tree: ~30% sensors/computing,
    #       ~70% drive trains, housings, mechanical structure.
    #       Reference total (8000 EUR at 60m²) from performance model step 0.
    #       TODO: Verify with supplier quotes for actuators; value likely too low
    #       for prototype hardware (IEA Task 48 estimates 15-30 kEUR for prototypes).
    # TODO: beta = 1.0 (linear with area) assumes actuator force ∝ tether force ∝ S.
    #       Range: beta=0.5 (Grete 2014, avionics dominated) to beta=1.5 (torque
    #       scales with force × lever arm ∝ S^1.5). Use beta=1.0 as default;
    #       perform sensitivity analysis for large kite areas (>100 m²).
    s_ref_avio = costs.avionicsReferenceArea
    beta = costs.avionicsScalingExponent
    avionics_capex = (costs.avionicsCostFixed
                      + costs.avionicsCostVarRef
                      * (kite.flatArea / s_ref_avio) ** beta)
else:
    # Fallback: fixed cost (Joshi & Trevisi 2024)
    avionics_capex = costs.avionicsCost
eco['avionics'] = {'CAPEX': avionics_capex, 'OPEX': 0}
```

### `src/ecomo/ecomo/loaders/eco_load_cost_inputs.py`

Load the new avionics fields alongside `avionicsCost`:

```python
avionics = _require(kite, 'avionics', 'costs.kite')
fields['avionicsCost'] = avionics['cost']
if 'cost_fixed' in avionics:
    fields['avionicsCostFixed'] = avionics['cost_fixed']
    fields['avionicsCostVarRef'] = avionics['cost_var_ref']
    fields['avionicsReferenceArea'] = avionics['reference_area']
    fields['avionicsScalingExponent'] = avionics.get('scaling_exponent', 1.0)
```

### `config/test/gg_soft/economic_cost_inputs_GG_soft.yml`

Update avionics section:

```yaml
    avionics:
      cost: 30000.0  # fallback fixed cost [EUR] (kept for backward compat.)
      # Scaled KCU model — preferred for soft-wing sizing studies
      # TODO: cost_fixed = 2400 EUR and cost_var_ref = 5600 EUR are rough estimates
      #       derived from the Braun (2015) KCU product tree (30/70 fixed/variable
      #       split) and performance model step-0 KCU cost of 8000 EUR at 25m².
      #       These values should be verified with supplier quotes for servo motors,
      #       gearboxes, and electronic controllers before publication.
      cost_fixed: 2400.0      # [EUR] fixed electronics/sensors/computing
      cost_var_ref: 5600.0    # [EUR] variable actuators at reference area
      reference_area: 60.0    # [m²]  S_ref for KCU scaling (same as kite cost model)
      scaling_exponent: 1.0   # [-]   beta; range 0.5–1.5
```

---

## Change 4 — Foundation (BoS): use peak mechanical power

**Why:** `eco_bos.py` currently sizes the foundation on `ratedPower` (rated electrical
output). The paper and Joshi & Trevisi (2024) Eq. 38 specify P_peak (peak mechanical
load). For the soft-wing power curve, P_peak ≈ 40 kW vs P_rated ≈ 13.7 kW — a factor
of ~2.9×. The generator and gearbox in `eco_gstation.py` already use
`peakMechanicalPower` correctly; the BoS is the only place this was missed.

### `src/ecomo/ecomo/subsystems/eco_bos.py`

```python
def eco_bos(performance, costs):
    ratedPowerKw = performance.ratedPower / W_PER_KW
    # Foundation is sized by peak mechanical load, not rated electrical power.
    # Joshi & Trevisi (2024) Eq. 38: C_found = p_found × P_peak.
    # For soft-wing pumping kites P_peak ≈ 2.9 × P_rated (vs. 2.5× assumed for
    # fixed-wing); use actual value from power curves when available.
    if performance.peakMechanicalPower is not None:
        peakPowerKw = performance.peakMechanicalPower / W_PER_KW
    else:
        peakPowerKw = ratedPowerKw  # fallback if peak power unavailable

    installCapex = costs.installationPricePower * ratedPowerKw
    return {
        'sitePrep': {'CAPEX': costs.sitePreparationPricePower * ratedPowerKw},
        'found':    {'CAPEX': costs.foundationPricePower * peakPowerKw},
        'install':  {'CAPEX': installCapex},
        'OM':       {'OPEX': costs.operationsMaintenancePricePower * ratedPowerKw},
        'decomm':   {'CAPEX': costs.decommissioningInstallationFraction * installCapex},
    }
```

---

## Change 5 — Drivetrain efficiencies

**Why:** All efficiencies in the system YAML are currently 1.0, so electrical power
equals mechanical power. The InertiaFree QSM already has the correct code structure:

```python
electrical_power = np.where(
    p >= 0.0,
    p * generator_efficiency,
    p / (motor_efficiency * storage_efficiency),
)
```

Note: `motor_efficiency` is read from the same `generator` YAML field (known code
behaviour). Using the same efficiency for generator and motor is physically reasonable
for PMSM machines operated in both modes and is an acceptable approximation at this
stage of analysis.

**File to change:** The system YAML used by InertiaFree QSM / AWESPA
(wherever `components.ground_station.generators` and `storages` are defined).

Update:

```yaml
components:
  ground_station:
    generators:
      - type: permanent_magnet_synchronous
        max_power: 40000.0
        # TODO: efficiency = 0.95 for both generator (reel-out) and motor (reel-in).
        #       Both use the same field due to current QSM code behaviour
        #       (motor_efficiency reads from generator entry — known limitation).
        #       Typical PMSM efficiency: 0.93–0.97 at rated load.
        #       TODO: Add separate motor_efficiency field to QSM when more detailed
        #       drivetrain data become available.
        #       The storage efficiency of 0.95 applies to the battery converter only
        #       (not double-counted: the existing 0.9 in the power module covers
        #       something else in the signal chain — verify before changing).
        efficiency: 0.95
    storages:
      - type: battery_bank
        # TODO: storage efficiency = 0.95 is a round-trip battery efficiency estimate.
        #       Typical Li-ion: 0.92–0.97. Update when battery specification is confirmed.
        efficiency: 0.95
```

**Important:** Before setting these values, confirm exactly where the existing η=0.9 is
applied in the power module call chain to avoid double-counting. Run a quick check:
compare `performance.electrical_power.average_cycle_power` with
`performance.power.average_cycle_power` in the AWESPA output at both η=1.0 and after
the change to confirm the expected ~5% reduction in AEP.

---

## Change 6 — Sensor cost placeholder update

**Why:** The current avionics cost of €30,000 is a rough prototype estimate. Sensor
hardware alone (Trimble BX982 GNSS + Xsens MTi-G IMU) costs ~€7,000–12,000 at unit
quantities. Since sensor cost is fixed (independent of kite size), it does not affect
scaling trends but does affect absolute LCoE level.

This is already captured in the `avionicsCostFixed` field added in Change 3. Add a
comment in the YAML:

```yaml
      # TODO: avionics cost_fixed = 2400 EUR is a rough placeholder.
      #       Sensor hardware (Trimble BX982 GNSS ≈ 6000 EUR + Xsens MTi-G ≈ 3000 EUR)
      #       already exceeds this value. Expert quote required before publication.
      #       Fixed cost — does not affect scaling trends, only absolute LCoE level.
```

---

## Summary of files changed

| File | Change |
|------|--------|
| `src/ecomo/ecomo/eco_costs.py` | Add `materialCostRef`, `referenceArea`, `materialScalingExponent`, `labourCostCoefficient`, `avionicsCostFixed`, `avionicsCostVarRef`, `avionicsReferenceArea`, `avionicsScalingExponent` to `KiteCosts` |
| `src/ecomo/ecomo/eco_inputs.py` | Add `reelOutTimeFraction: Optional[np.ndarray]` to `PerformanceData` |
| `src/ecomo/ecomo/subsystems/eco_kite.py` | New two-term kite cost formula; time-weighted LF; area-scaled avionics |
| `src/ecomo/ecomo/subsystems/eco_bos.py` | Foundation uses `peakMechanicalPower` instead of `ratedPower` |
| `src/ecomo/ecomo/loaders/eco_load_cost_inputs.py` | Load new kite cost and avionics fields |
| `src/ecomo/ecomo_economic.py` | Extract `reelOutFraction` and `peakReelOutPower` from power curves; pass to `PerformanceData`; remove 2.5× fallback in AWESPA path |
| `config/test/gg_soft/economic_cost_inputs_GG_soft.yml` | New kite cost and avionics parameters; updated kite lifetime |
| AWESPA system YAML | Set `efficiency: 0.95` for generator and storage |

---

## Verification steps after implementation

1. Run the existing test suite: `pytest tests/` — all tests must pass.
2. Run the GG soft-wing example and confirm:
   - Kite CAPEX at S=19.75 m² ≈ €1,250–1,700 (consistent with V3 historical price ~€2,000
     as upper bound; model gives lean design estimate so lower values are expected)
   - Kite OPEX (replacement) dominates over kite CAPEX (due to 250h lifetime)
   - Foundation CAPEX is ~2.9× larger than with the old `ratedPower` sizing
   - LF_time_weighted ≈ 0.219 (vs old unweighted 0.355); check `eco['kite']['structure']['f_repl']`
3. Confirm no double-counting of efficiencies by comparing AEP before and after
   efficiency change (expect ~5% reduction per efficiency stage applied).
4. Check that changing `material_scaling_exponent` from 1.00 to 1.22 shifts predicted
   optimal kite area — this validates the exponent is propagating correctly into LCoE.

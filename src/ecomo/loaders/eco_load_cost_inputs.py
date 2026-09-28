"""Loader for the economic cost inputs.

Builds the :class:`EcoCosts` cost parameters from an
``economic_cost_inputs_*.yml`` file.
"""

import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import yaml

from ..constants import (
    COMPONENT_COST_MODEL_NAMES,
    DRIVETRAIN_NAMES,
    KITE_STRUCTURE_COST_MODEL_NAMES,
    STORAGE_NAMES,
    WINCH_MATERIAL_NAMES,
)
from ..eco_costs import (
    BalanceOfSystemCosts,
    EcoCosts,
    GearboxCosts,
    GeneratorCosts,
    GroundStationCosts,
    KiteCosts,
    MarketCosts,
    StorageCosts,
    TetherCosts,
    WinchCosts,
)


def _require(section: Dict[str, Any], key: str, context: str):
    """Read a required key from a YAML section.

    Args:
        section (dict): YAML section to read from.
        key (str): Required key.
        context (str): Dotted path of the section, used in the error.

    Raises:
        KeyError: If the key is missing.

    Returns:
        The value of the key.
    """
    if key not in section:
        raise KeyError(f"Missing required key '{context}.{key}' "
                       f"in cost inputs file")
    return section[key]


def _load_kite_costs(kite: Dict[str, Any], power: Optional[str]) -> KiteCosts:
    """Build the kite cost parameters from the YAML kite section."""
    structure = kite.get('structure', {})
    avionics = _require(kite, 'avionics', 'costs.kite')
    fields: Dict[str, Any] = {
        'avionicsCost': avionics['cost'],
        'avionicsLifetime': avionics.get('lifetime_years'),
    }
    if 'cost_fixed' in avionics:
        fields['avionicsCostFixed'] = avionics['cost_fixed']
        fields['avionicsCostVarRef'] = _require(
            avionics, 'cost_var_ref', 'costs.kite.avionics')
        fields['avionicsReferenceArea'] = _require(
            avionics, 'reference_area', 'costs.kite.avionics')
        fields['avionicsScalingExponent'] = avionics.get(
            'scaling_exponent', 1.0)

    if 'fixed' in structure:
        fixed = structure['fixed']
        massArea = fixed.get('mass_area', {})
        laminate = fixed.get('laminate', {})
        fields['structureCostModel'] = KITE_STRUCTURE_COST_MODEL_NAMES[
            _require(fixed, 'cost_model', 'costs.kite.structure.fixed')]
        fields['priceStructuralMass'] = massArea.get('price_mass')
        fields['priceWettedArea'] = massArea.get('price_wetted_area')
        fields['priceUniax'] = laminate.get('price_uniax')
        fields['priceTriax'] = laminate.get('price_triax')
        fields['manufacturingFactor'] = laminate.get('manufacturing_factor')
    if 'soft' in structure:
        soft = structure['soft']
        # Two-term scaling model (preferred) or flat-price model
        if 'material_cost_ref' in soft:
            fields['materialCostRef'] = soft['material_cost_ref']
            fields['referenceArea'] = _require(
                soft, 'reference_area', 'costs.kite.structure.soft')
            fields['materialScalingExponent'] = _require(
                soft, 'material_scaling_exponent',
                'costs.kite.structure.soft')
            fields['labourCostCoefficient'] = _require(
                soft, 'labour_cost_coefficient',
                'costs.kite.structure.soft')
        else:
            fields['priceFabric'] = _require(soft, 'price_fabric',
                                             'costs.kite.structure.soft')
            fields['priceBridle'] = _require(soft, 'price_bridle',
                                             'costs.kite.structure.soft')
        # Replacement model. The reel-out-hour model (preferred) uses the
        # canopy life in loaded hours; the legacy calendar model uses the
        # lifetime in flying years. Require one of the two.
        if 'canopy_lifetime_flight_hours' in soft:
            fields['canopyLifetimeFlightHours'] = (
                soft['canopy_lifetime_flight_hours'])
            fields['perCyclePenalty'] = soft.get('per_cycle_penalty')
            fields['canopyLoadExponent'] = soft.get('canopy_load_exponent')
            fields['canopyReferenceForce'] = soft.get('canopy_reference_force')
            fields['structureLifetime'] = soft.get('lifetime_flying_years')
        else:
            fields['structureLifetime'] = _require(
                soft, 'lifetime_flying_years', 'costs.kite.structure.soft')

    if 'sensor' in kite:
        fields['sensorCost'] = kite['sensor'].get('cost')
        fields['sensorLifetime'] = kite['sensor'].get('lifetime_years')

    if 'onboard_generator' in kite:
        fields['onboardGeneratorPricePower'] = (
            kite['onboard_generator']['price_power'])
    if 'onboard_battery' in kite:
        fields['onboardBatteryPriceEnergy'] = (
            kite['onboard_battery']['price_energy'])
    elif power == 'GG':
        warnings.warn(
            "'onboard_battery' not found in cost inputs file; it is "
            "needed for the ground-gen onboard battery cost. Set "
            "'costs.kite.onboard_battery' in the cost inputs file.",
            UserWarning, stacklevel=2,
        )

    return KiteCosts(**fields)


def _resolve_bending_life_a1(
    tether: Dict[str, Any],
    drum_to_tether_ratio: Optional[float],
) -> Optional[float]:
    """Resolve the bending fatigue coefficient a1 [-].

    An explicit scalar ``bending_life_a1`` always takes precedence
    (backward compatible). Otherwise, if ``bending_life_a1_table`` is
    given, a1 is interpolated -- piecewise log-linear in the drum-to-
    tether diameter ratio D/d, which matches how bend-fatigue S-N data
    is normally reported (life increases steeply, roughly log-linearly,
    with a gentler bend radius) -- at the winch's
    ``drum_to_tether_diameter_ratio``. This couples the winch design
    (D/d, which also sizes the winch CAPEX in ``eco_gstation``) to the
    tether bending life, so a smaller/cheaper drum is charged a shorter
    tether life. Extrapolating beyond the table's calibrated range is
    flagged with a warning (untested regime).

    Args:
        tether (dict): YAML ``costs.tether`` section.
        drum_to_tether_ratio (float): The winch's D/d [-], or None.

    Raises:
        KeyError: If a table is given but the winch's D/d ratio is not
            available to resolve it.
        ValueError: If the table's ratio/a1 arrays have different
            lengths.

    Returns:
        float: The resolved a1, or None if neither is configured.
    """
    if 'bending_life_a1' in tether:
        return tether['bending_life_a1']
    table = tether.get('bending_life_a1_table')
    if table is None:
        return None
    if drum_to_tether_ratio is None:
        raise KeyError(
            "'costs.tether.bending_life_a1_table' is set but "
            "'costs.ground_station.winch.drum_to_tether_diameter_ratio' "
            "is not; the bending fatigue coefficient a1 cannot be "
            "resolved from the D/d table without the winch's drum ratio."
        )
    ratios = np.asarray(table['drum_to_tether_ratio'], dtype=float)
    a1Values = np.asarray(table['a1'], dtype=float)
    if ratios.shape != a1Values.shape:
        raise ValueError(
            "'costs.tether.bending_life_a1_table': 'drum_to_tether_ratio' "
            f"({ratios.size} entries) and 'a1' ({a1Values.size} entries) "
            "must have the same length."
        )
    order = np.argsort(ratios)
    ratios, a1Values = ratios[order], a1Values[order]
    logRatios = np.log10(ratios)
    logQuery = np.log10(drum_to_tether_ratio)
    if not (logRatios[0] <= logQuery <= logRatios[-1]):
        warnings.warn(
            f"drum_to_tether_diameter_ratio={drum_to_tether_ratio:g} is "
            f"outside the calibrated bending_life_a1_table range "
            f"[{ratios[0]:g}, {ratios[-1]:g}]; a1 is extrapolated "
            "(log-linear) and untested outside this range.",
            UserWarning, stacklevel=2,
        )
    return float(np.interp(logQuery, logRatios, a1Values))


def _resolve_operational_life(tether: Dict[str, Any]) -> Optional[float]:
    """Resolve the effective operational tether life [h], or None.

    The empirical operational-wear mode is opt-in: it is only active when
    ``operational_life_enabled`` is true (default false). When disabled,
    None is returned so the tether life is governed by bending/creep
    fatigue alone. The ``operational_life_flight_hours`` value is kept in
    the file either way, so the mode can be toggled without re-entering
    the number.

    Args:
        tether (dict): YAML ``costs.tether`` section.

    Returns:
        float: The operational life in flight hours when enabled, else
        None (mode off).
    """
    if not tether.get('operational_life_enabled', False):
        return None
    return _require(tether, 'operational_life_flight_hours',
                    'costs.tether (operational_life_enabled is true)')


def _load_tether_costs(tether: Dict[str, Any],
                       max_stress_override: Optional[float],
                       drum_to_tether_ratio: Optional[float] = None,
                       ) -> TetherCosts:
    """Build the tether cost parameters from the YAML tether section."""
    maxStress = (max_stress_override if max_stress_override is not None
                 else _require(tether, 'max_stress', 'costs.tether'))
    return TetherCosts(
        priceMass=_require(tether, 'price_mass', 'costs.tether'),
        fibreAreaFraction=_require(tether, 'fibre_area_fraction',
                                   'costs.tether'),
        coatingMassFraction=_require(tether, 'coating_mass_fraction',
                                     'costs.tether'),
        maxStress=maxStress,
        creepLifeCoefficients=np.asarray(
            _require(tether, 'creep_life_coefficients', 'costs.tether'),
            dtype=float),
        conductiveManufacturingFactor=tether.get(
            'conductive_manufacturing_factor'),
        bendingLifeA1=_resolve_bending_life_a1(tether, drum_to_tether_ratio),
        bendingLifeA2=tether.get('bending_life_a2'),
        nBends=tether.get('n_bends'),
        operationalLife=_resolve_operational_life(tether),
        **_master_curve_fields(tether, drum_to_tether_ratio),
    )


def _master_curve_fields(tether: Dict[str, Any],
                         drum_to_tether_ratio: Optional[float]) -> Dict[str, Any]:
    """Read the optional Meuwissen/Bosman bending master-curve block.

    Returns the TetherCosts master-curve fields (all None when the block
    is absent, so the a1/a2 fallback stays active). When the block is
    present, the winch D/d ratio is required to evaluate the bearing
    pressure.

    Args:
        tether (dict): YAML ``costs.tether`` section.
        drum_to_tether_ratio (float): The winch's D/d [-], or None.

    Raises:
        KeyError: If the master curve is configured but the winch D/d is
            not available.
    """
    mc = tether.get('master_curve')
    if mc is None:
        return {}
    if drum_to_tether_ratio is None:
        raise KeyError(
            "'costs.tether.master_curve' is set but "
            "'costs.ground_station.winch.drum_to_tether_diameter_ratio' "
            "is not; the bearing-pressure master curve needs the winch D/d."
        )
    return {
        'masterCurveCoeff': mc.get('coefficient'),
        'masterCurveExponent': mc.get('exponent'),
        'bearingPressureCoeff': mc.get('bearing_pressure_coeff'),
        'pwLimitMpa': mc.get('pw_limit_mpa'),
        'designSafetyFactor': mc.get('design_safety_factor'),
        'bendingDdRatio': drum_to_tether_ratio,
    }


def _load_gstation_costs(gs: Dict[str, Any]) -> GroundStationCosts:
    """Build the ground station cost parameters from its YAML section."""
    winchRaw = _require(gs, 'winch', 'costs.ground_station')
    materials = winchRaw.get('materials', {})
    aluminum = materials.get('aluminum', {})
    steel = materials.get('steel', {})
    winch = WinchCosts(
        material=WINCH_MATERIAL_NAMES[
            _require(winchRaw, 'material', 'costs.ground_station.winch')],
        drumToTetherDiameterRatio=winchRaw.get(
            'drum_to_tether_diameter_ratio'),
        safetyFactorDiameter=winchRaw.get('safety_factor_diameter'),
        safetyFactorLength=winchRaw.get('safety_factor_length'),
        priceAluminum=aluminum.get('price_mass'),
        densityAluminum=aluminum.get('density'),
        maxStressAluminum=aluminum.get('max_stress'),
        priceSteel=steel.get('price_mass'),
        densitySteel=steel.get('density'),
        maxStressSteel=steel.get('max_stress'),
        lifetime=winchRaw.get('lifetime_years'),
    )

    fields: Dict[str, Any] = {
        'winch': winch,
        'electricalStorage': STORAGE_NAMES[
            _require(gs, 'electrical_storage', 'costs.ground_station')],
        'ultracapacitor': StorageCosts(
            priceEnergy=_require(gs, 'ultracapacitor',
                                 'costs.ground_station')['price_energy'],
            cycleLife=gs['ultracapacitor']['cycle_life'],
        ),
        'battery': StorageCosts(
            priceEnergy=_require(gs, 'battery',
                                 'costs.ground_station')['price_energy'],
            cycleLife=gs['battery']['cycle_life'],
        ),
        'powerConverterPricePower': _require(
            gs, 'power_converter', 'costs.ground_station')['price_power'],
        'powerConverterLifetime': gs['power_converter'].get('lifetime_years'),
    }

    if 'drivetrain' in gs:
        fields['drivetrain'] = DRIVETRAIN_NAMES[gs['drivetrain']]
    if 'gearbox' in gs:
        gearbox = gs['gearbox']
        massBased = gearbox.get('mass_based', {})
        fields['gearbox'] = GearboxCosts(
            costModel=COMPONENT_COST_MODEL_NAMES[gearbox['cost_model']],
            pricePower=gearbox.get('power_based', {}).get('price_power'),
            priceMass=massBased.get('price_mass'),
            massCoefficient=massBased.get('mass_coefficient'),
            massExponent=massBased.get('mass_exponent'),
            lifetime=gearbox.get('lifetime_years'),
        )
    if 'generator' in gs:
        generator = gs['generator']
        massBased = generator.get('mass_based', {})
        fields['generator'] = GeneratorCosts(
            costModel=COMPONENT_COST_MODEL_NAMES[generator['cost_model']],
            pricePower=generator.get('power_based', {}).get('price_power'),
            priceMass=massBased.get('price_mass'),
            massSlope=massBased.get('mass_slope'),
            massOffset=massBased.get('mass_offset'),
            lifetime=generator.get('lifetime_years'),
        )
    isHydraulic = gs.get('drivetrain') == 'hydraulic'
    if 'pump_motor' in gs:
        fields['pumpMotorPricePower'] = gs['pump_motor']['price_power']
        fields['pumpMotorMaintenancePricePower'] = (
            gs['pump_motor']['maintenance_price_power'])
    elif isHydraulic:
        _warn_missing_hydraulic('pump_motor')
    if 'hydraulic_accumulator' in gs:
        fields['hydraulicAccumulatorPriceEnergy'] = (
            gs['hydraulic_accumulator']['price_energy'])
        fields['hydraulicAccumulatorMaintenancePriceEnergy'] = (
            gs['hydraulic_accumulator']['maintenance_price_energy'])
    elif isHydraulic:
        _warn_missing_hydraulic('hydraulic_accumulator')
    if 'hydraulic_motor' in gs:
        fields['hydraulicMotorPricePower'] = (
            gs['hydraulic_motor']['price_power'])
        fields['hydraulicMotorMaintenancePricePower'] = (
            gs['hydraulic_motor']['maintenance_price_power'])
    elif isHydraulic:
        _warn_missing_hydraulic('hydraulic_motor')

    if 'launch_land' in gs:
        fields['launchLandCost'] = gs['launch_land'].get('cost')
        fields['launchLandPriceArea'] = gs['launch_land'].get('price_area')
        fields['launchLandLifetime'] = gs['launch_land'].get('lifetime_years')

    return GroundStationCosts(**fields)


def _warn_missing_hydraulic(component: str) -> None:
    """Warn that a hydraulic-drivetrain component cost is absent."""
    warnings.warn(
        f"'{component}' not found in cost inputs file; it is needed for "
        "the hydraulic drivetrain. Set "
        f"'costs.ground_station.{component}' in the cost inputs file.",
        UserWarning, stacklevel=2,
    )


def _apply_stage_defaults_to_costs(
        data: Dict[str, Any],
        stage_defaults: Optional[Dict[str, float]]) -> None:
    """Apply the development-stage maturity settings to the cost inputs.

    Mutates ``data`` in place. The canopy life and the tether
    operational life are filled only where they are not already present
    (explicit numeric file values win). Selecting a stage additionally
    forces the tether into operational-wear mode -- operational life
    enabled, bending ``master_curve`` off -- so the staged operational
    life governs the replacement. An explicit
    ``costs.tether.operational_life_enabled: false`` opts out and keeps the
    bending master curve, so the design study runs a stress-sensitive tether
    life at a chosen maturity; otherwise the stage's operational mode wins.
    No-op when ``stage_defaults`` is empty (stage ``none``).

    Args:
        data (dict): Parsed cost inputs mapping.
        stage_defaults (dict): Development-stage defaults, or None.
    """
    if not stage_defaults:
        return
    soft = (data.get('costs', {}).get('kite', {})
            .get('structure', {}).get('soft'))
    if soft is not None and 'canopy_lifetime_flight_hours' not in soft:
        soft['canopy_lifetime_flight_hours'] = (
            stage_defaults['canopy_lifetime_flight_hours'])
    tether = data.get('costs', {}).get('tether')
    if tether is not None:
        if 'operational_life_flight_hours' not in tether:
            tether['operational_life_flight_hours'] = (
                stage_defaults['operational_life_flight_hours'])
        # A stage models maturity via operational wear, not the bending
        # design safety factor: by default it puts the tether into
        # operational-life mode (operational life governs, master curve off).
        # An explicit ``operational_life_enabled: false`` in the cost file is
        # an opt-out that wins, so the design study can keep the bending master
        # curve -- and hence a stress-sensitive tether life -- while still
        # taking canopy life, operating hours and maintenance from the stage.
        if tether.get('operational_life_enabled', True):
            tether['operational_life_enabled'] = True
            tether['master_curve'] = None


def eco_load_cost_inputs(cost_inputs_path: Path,
                         tether_max_stress: Optional[float] = None,
                         power: Optional[str] = None,
                         stage_defaults: Optional[Dict[str, float]] = None
                         ) -> EcoCosts:
    """Load cost model parameters from a YAML file.

    Args:
        cost_inputs_path (Path): Path to an
            ``economic_cost_inputs_*.yml`` file.
        tether_max_stress (float): Maximum tether stress [Pa] to use
            instead of the value in the cost file (e.g. the material
            breaking strength from the system YAML). Defaults to None.
        power (str): Power generation type, 'GG' or 'FG', used to decide
            which topology-dependent cost fields warrant a warning when
            absent. Defaults to None (no topology-dependent warnings).
        stage_defaults (dict): Development-stage maturity defaults used to
            fill the canopy life and tether operational life when they are
            absent from the file. Explicit file values take precedence.
            Defaults to None.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If a required section or key is missing.

    Returns:
        EcoCosts: The cost parameter structure.
    """
    costPath = Path(cost_inputs_path)
    if not costPath.exists():
        raise FileNotFoundError(
            f"Cost inputs file not found: {costPath}")

    with open(costPath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    _apply_stage_defaults_to_costs(data, stage_defaults)

    costs = _require(data, 'costs', '<root>')
    market = _require(data, 'market', '<root>')
    bos = _require(costs, 'balance_of_system', 'costs')

    # The winch's D/d ratio is read here (ahead of the ground station
    # cost loading below) because it couples into the tether bending
    # life when 'bending_life_a1_table' is used (see
    # _resolve_bending_life_a1).
    groundStationRaw = _require(costs, 'ground_station', 'costs')
    drumToTetherRatio = groundStationRaw.get('winch', {}).get(
        'drum_to_tether_diameter_ratio')

    return EcoCosts(
        kite=_load_kite_costs(_require(costs, 'kite', 'costs'), power),
        tether=_load_tether_costs(_require(costs, 'tether', 'costs'),
                                  tether_max_stress, drumToTetherRatio),
        groundStation=_load_gstation_costs(groundStationRaw),
        balanceOfSystem=BalanceOfSystemCosts(
            sitePreparationPricePower=bos['site_preparation']['price_power'],
            foundationPricePower=bos['foundation']['price_power'],
            installationPricePower=bos['installation']['price_power'],
            operationsMaintenancePricePower=(
                bos['operations_maintenance']['price_power']),
            decommissioningInstallationFraction=(
                bos['decommissioning']['installation_fraction']),
        ),
        market=MarketCosts(
            electricityPriceIntercept=market['electricity_price']['intercept'],
            electricityPriceSlope=market['electricity_price'][
                'wind_speed_slope'],
            subsidy=market['subsidy'],
        ),
    )

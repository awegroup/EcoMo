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
    fields: Dict[str, Any] = {
        'avionicsCost': _require(kite, 'avionics', 'costs.kite')['cost'],
    }

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
        fields['priceFabric'] = _require(soft, 'price_fabric',
                                         'costs.kite.structure.soft')
        fields['priceBridle'] = _require(soft, 'price_bridle',
                                         'costs.kite.structure.soft')
        fields['structureLifetime'] = _require(soft, 'lifetime_flying_years',
                                               'costs.kite.structure.soft')

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


def _load_tether_costs(tether: Dict[str, Any],
                       max_stress_override: Optional[float]) -> TetherCosts:
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
        bendingLifeA1=tether.get('bending_life_a1'),
        bendingLifeA2=tether.get('bending_life_a2'),
        nBends=tether.get('n_bends'),
    )


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

    return GroundStationCosts(**fields)


def _warn_missing_hydraulic(component: str) -> None:
    """Warn that a hydraulic-drivetrain component cost is absent."""
    warnings.warn(
        f"'{component}' not found in cost inputs file; it is needed for "
        "the hydraulic drivetrain. Set "
        f"'costs.ground_station.{component}' in the cost inputs file.",
        UserWarning, stacklevel=2,
    )


def eco_load_cost_inputs(cost_inputs_path: Path,
                         tether_max_stress: Optional[float] = None,
                         power: Optional[str] = None
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

    costs = _require(data, 'costs', '<root>')
    market = _require(data, 'market', '<root>')
    bos = _require(costs, 'balance_of_system', 'costs')

    return EcoCosts(
        kite=_load_kite_costs(_require(costs, 'kite', 'costs'), power),
        tether=_load_tether_costs(_require(costs, 'tether', 'costs'),
                                  tether_max_stress),
        groundStation=_load_gstation_costs(
            _require(costs, 'ground_station', 'costs')),
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

"""Load physical system parameters from an awesIO system YAML file.

Extracts the parameters the economic model needs (wing mass and area,
tether geometry, storage capacities) from a system configuration file
in awesIO ``system_schema.yml`` format.

The cost model needs the flat wing area. When the system file provides
a ``flat_wing_area`` field it is used directly; otherwise the flat
area of soft kites is estimated from the projected area with the
factor 25/18 (Joshi & Trevisi 2024, section 2.1.2).
"""

import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

try:
    from awesio.validator import validate as awesio_validate  # type: ignore
except ImportError:
    awesio_validate = None

# Flat-to-projected wing area ratio for soft kites
FLAT_TO_PROJECTED_AREA = 25 / 18
WH_PER_KWH = 1e3


def eco_load_system(system_path: Path, wing: str,
                    validate: bool = True) -> Dict[str, Any]:
    """Load system parameters from an awesIO system YAML file.

    Args:
        system_path (Path): Path to a system configuration YAML file
            (awesIO ``system_schema.yml`` format).
        wing (str): Wing type, 'fixed' or 'soft', used for the flat
            wing area estimation.
        validate (bool): If True, validate the file with the awesIO
            validator. Defaults to True.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If required components are missing.

    Returns:
        dict: System parameters with keys ``'kiteMass'`` [kg],
        ``'kiteFlatArea'`` [m2], ``'kiteSpan'`` [m or None],
        ``'tether'`` (dict with d/L/rho), ``'tetherMaxStress'``
        [Pa or None] and ``'storageCapacities'`` [kWh per storage type
        name].
    """
    systemPath = Path(system_path)
    if not systemPath.exists():
        raise FileNotFoundError(f"System file not found: {systemPath}")

    with open(systemPath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if validate and awesio_validate is not None:
        awesio_validate(data)

    components = data['components']
    wingStructure = components['kites'][0]['wing']['structure']
    tetherStructure = components['tethers'][0]['structure']
    groundStation = components['ground_station']

    # --- kite: resolve the flat wing area used by the cost model -------
    span = wingStructure.get('span')
    aspectRatio = wingStructure.get('aspect_ratio')
    if 'flat_wing_area' in wingStructure:
        flatArea = wingStructure['flat_wing_area']
    elif span is not None and aspectRatio is not None:
        # Exact geometric relation for fixed wings, not a fallback
        flatArea = span ** 2 / aspectRatio
    elif wing == 'soft':
        warnings.warn(
            "'flat_wing_area' not found in system YAML; estimating from "
            "projected_surface_area using the 25/18 soft-kite conversion "
            "factor (Joshi & Trevisi 2024, section 2.1.2). Add "
            "'flat_wing_area' to the system YAML to use the exact value.",
            UserWarning, stacklevel=2,
        )
        flatArea = (FLAT_TO_PROJECTED_AREA *
                    wingStructure['projected_surface_area'])
    else:
        warnings.warn(
            "'flat_wing_area' not found in system YAML; using "
            "projected_surface_area as the flat wing area. Add "
            "'flat_wing_area' to the system YAML to use the exact value.",
            UserWarning, stacklevel=2,
        )
        flatArea = wingStructure['projected_surface_area']

    # --- tether --------------------------------------------------------
    tether = {
        'd': tetherStructure['diameter'],
        'L': tetherStructure['length'],
        'rho': tetherStructure['density'],
    }
    tetherMaxStress: Optional[float] = (
        tetherStructure.get('material', {}).get('breaking_strength'))
    if tetherMaxStress is None:
        warnings.warn(
            "Tether 'material.breaking_strength' not found in system "
            "YAML; using 'max_stress' from the cost inputs file as "
            "fallback.",
            UserWarning, stacklevel=2,
        )

    # --- ground station storages ----------------------------------------
    storageCapacities: Dict[str, float] = {}
    typeNames = {'capacitor_bank': 'ultracapacitor',
                 'battery_bank': 'battery'}
    for storage in groundStation.get('storages', []):
        name = typeNames.get(storage.get('type'))
        if name is not None:
            storageCapacities[name] = storage['capacity'] / WH_PER_KWH

    return {
        'kiteMass': wingStructure['mass'],
        'kiteFlatArea': flatArea,
        'kiteSpan': span,
        'tether': tether,
        'tetherMaxStress': tetherMaxStress,
        'storageCapacities': storageCapacities,
    }

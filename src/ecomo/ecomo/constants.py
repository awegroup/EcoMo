"""Shared constants for the ECOMo subsystem modules.

Defines unit conversion factors and the enumerations for the integer
codes used in the cost input Excel files.
"""

from enum import IntEnum

# Time
HOURS_PER_YEAR = 8760
SECONDS_PER_HOUR = 3600

# Unit conversions
W_PER_KW = 1e3
W_PER_MW = 1e6
KWH_PER_MWH = 1e3
PA_PER_GPA = 1e9


class WinchMaterial(IntEnum):
    """Winch drum material code (cost inputs: gStation.winch.material)."""

    ALUMINUM = 1
    STEEL = 2


class DrivetrainType(IntEnum):
    """Drivetrain type code (cost inputs: gStation.drivetrain_type)."""

    ELECTRIC = 1
    HYDRAULIC = 2


class StorageType(IntEnum):
    """Electrical storage code (cost inputs: gStation.elecSto_type)."""

    ULTRACAPACITOR = 1
    BATTERY = 2


class KiteStructureCostModel(IntEnum):
    """Fixed-wing kite structure cost model.

    MASS_AREA scales with structural mass and wetted area; LAMINATE
    scales with the uniax/triax laminate masses.
    """

    MASS_AREA = 1
    LAMINATE = 2


class ComponentCostModel(IntEnum):
    """Gearbox/generator cost model.

    POWER_BASED scales the cost with rated power; MASS_BASED scales it
    with an estimated component mass (WISDEM-style fits).
    """

    POWER_BASED = 1
    MASS_BASED = 2


# String names used in the YAML input files for the enum codes
WINCH_MATERIAL_NAMES = {
    'aluminum': WinchMaterial.ALUMINUM,
    'steel': WinchMaterial.STEEL,
}
DRIVETRAIN_NAMES = {
    'electric': DrivetrainType.ELECTRIC,
    'hydraulic': DrivetrainType.HYDRAULIC,
}
STORAGE_NAMES = {
    'ultracapacitor': StorageType.ULTRACAPACITOR,
    'battery': StorageType.BATTERY,
}
KITE_STRUCTURE_COST_MODEL_NAMES = {
    'mass_area': KiteStructureCostModel.MASS_AREA,
    'laminate': KiteStructureCostModel.LAMINATE,
}
COMPONENT_COST_MODEL_NAMES = {
    'power_based': ComponentCostModel.POWER_BASED,
    'mass_based': ComponentCostModel.MASS_BASED,
}

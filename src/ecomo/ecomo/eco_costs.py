"""Cost parameter structures for the ECOMo simulation.

The frozen dataclasses in this module hold the cost model parameters
of one simulation, grouped per subsystem.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constants import (
    ComponentCostModel,
    DrivetrainType,
    KiteStructureCostModel,
    StorageType,
    WinchMaterial,
)


@dataclass(frozen=True)
class KiteCosts:
    """Kite subsystem cost parameters.

    Attributes:
        avionicsCost: Fixed avionics cost [EUR].
        structureCostModel: Fixed-wing structure cost model, or None
            for soft wings.
        priceStructuralMass: Structural mass price [EUR/kg] (mass-area
            model).
        priceWettedArea: Wetted area price [EUR/m2] (mass-area model).
        priceUniax: Uniax laminate price [EUR/kg] (laminate model).
        priceTriax: Triax laminate price [EUR/kg] (laminate model).
        manufacturingFactor: Manufacturing cost factor [-] (laminate
            model).
        priceFabric: Soft-wing fabric price [EUR/m2].
        priceBridle: Soft-wing bridle price [EUR/m2].
        structureLifetime: Soft-wing structural lifetime at full
            loading [flying years].
        onboardGeneratorPricePower: Onboard generator price [EUR/kW],
            or None.
        onboardBatteryPriceEnergy: Onboard battery price [EUR/kWh], or
            None.
    """

    avionicsCost: float
    structureCostModel: Optional[KiteStructureCostModel] = None
    priceStructuralMass: Optional[float] = None
    priceWettedArea: Optional[float] = None
    priceUniax: Optional[float] = None
    priceTriax: Optional[float] = None
    manufacturingFactor: Optional[float] = None
    priceFabric: Optional[float] = None
    priceBridle: Optional[float] = None
    structureLifetime: Optional[float] = None
    onboardGeneratorPricePower: Optional[float] = None
    onboardBatteryPriceEnergy: Optional[float] = None


@dataclass(frozen=True)
class TetherCosts:
    """Tether subsystem cost parameters.

    Attributes:
        priceMass: Tether price [EUR/kg].
        fibreAreaFraction: Fibre to total cross-sectional area ratio
            [-].
        coatingMassFraction: Coating mass fraction of the total tether
            mass [-].
        maxStress: Maximum allowable fibre stress [Pa].
        creepLifeCoefficients: Polynomial coefficients of the creep
            life curve.
        conductiveManufacturingFactor: Manufacturing factor for
            conductive (FG) tethers [-], or None.
        bendingLifeA1: Bending fatigue coefficient a1 [-], or None.
        bendingLifeA2: Bending fatigue coefficient a2 [-], or None.
        nBends: Number of bends (pulleys) per cycle [-], or None.
    """

    priceMass: float
    fibreAreaFraction: float
    coatingMassFraction: float
    maxStress: float
    creepLifeCoefficients: np.ndarray
    conductiveManufacturingFactor: Optional[float] = None
    bendingLifeA1: Optional[float] = None
    bendingLifeA2: Optional[float] = None
    nBends: Optional[float] = None


@dataclass(frozen=True)
class WinchCosts:
    """Winch cost parameters.

    Attributes:
        material: Winch drum material.
        drumToTetherDiameterRatio: Winch drum to tether diameter ratio
            [-].
        safetyFactorDiameter: Safety margin on the tether diameter [-].
        safetyFactorLength: Safety margin on the tether length [-].
        priceAluminum: Aluminum price [EUR/kg].
        densityAluminum: Aluminum density [kg/m3].
        maxStressAluminum: Aluminum strength [Pa].
        priceSteel: Steel price [EUR/kg].
        densitySteel: Steel density [kg/m3].
        maxStressSteel: Steel strength [Pa].
    """

    material: WinchMaterial
    drumToTetherDiameterRatio: float
    safetyFactorDiameter: float
    safetyFactorLength: float
    priceAluminum: float
    densityAluminum: float
    maxStressAluminum: float
    priceSteel: float
    densitySteel: float
    maxStressSteel: float

    @property
    def price(self) -> float:
        """Price of the selected winch material [EUR/kg]."""
        return (self.priceAluminum if self.material == WinchMaterial.ALUMINUM
                else self.priceSteel)

    @property
    def density(self) -> float:
        """Density of the selected winch material [kg/m3]."""
        return (self.densityAluminum if self.material == WinchMaterial.ALUMINUM
                else self.densitySteel)

    @property
    def maxStress(self) -> float:
        """Strength of the selected winch material [Pa]."""
        return (self.maxStressAluminum
                if self.material == WinchMaterial.ALUMINUM
                else self.maxStressSteel)


@dataclass(frozen=True)
class GearboxCosts:
    """Gearbox cost parameters.

    Attributes:
        costModel: Gearbox cost model.
        pricePower: Power-based price [EUR/kW] (power-based model).
        priceMass: Mass-based price [EUR/kg] (mass-based model).
        massCoefficient: Mass scaling coefficient (mass-based model).
        massExponent: Mass scaling exponent (mass-based model).
    """

    costModel: ComponentCostModel
    pricePower: Optional[float] = None
    priceMass: Optional[float] = None
    massCoefficient: Optional[float] = None
    massExponent: Optional[float] = None


@dataclass(frozen=True)
class GeneratorCosts:
    """Electric generator cost parameters.

    Attributes:
        costModel: Generator cost model.
        pricePower: Power-based price [EUR/kW] (power-based model).
        priceMass: Mass-based price [EUR/kg] (mass-based model).
        massSlope: Mass scaling slope (mass-based model).
        massOffset: Mass offset (mass-based model).
    """

    costModel: ComponentCostModel
    pricePower: Optional[float] = None
    priceMass: Optional[float] = None
    massSlope: Optional[float] = None
    massOffset: Optional[float] = None


@dataclass(frozen=True)
class StorageCosts:
    """Energy storage cost parameters.

    Attributes:
        priceEnergy: Storage price [EUR/kWh].
        cycleLife: Rated number of charge/discharge cycles [-].
    """

    priceEnergy: float
    cycleLife: float


@dataclass(frozen=True)
class GroundStationCosts:
    """Ground station subsystem cost parameters.

    Attributes:
        winch: Winch cost parameters.
        electricalStorage: Selected electrical storage type.
        ultracapacitor: Ultracapacitor cost parameters.
        battery: Battery cost parameters.
        powerConverterPricePower: Power converter price [EUR/kW].
        drivetrain: Drivetrain type (GG systems), or None.
        gearbox: Gearbox cost parameters (electric drivetrain), or
            None.
        generator: Generator cost parameters (GG electric/hydraulic),
            or None.
        pumpMotorPricePower: Pump-motor price [EUR/kW], or None.
        pumpMotorMaintenancePricePower: Pump-motor maintenance price
            [EUR/kW], or None.
        hydraulicAccumulatorPriceEnergy: Accumulator price [EUR/kWh],
            or None.
        hydraulicAccumulatorMaintenancePriceEnergy: Accumulator
            maintenance price [EUR/kWh], or None.
        hydraulicMotorPricePower: Hydraulic motor price [EUR/kW], or
            None.
        hydraulicMotorMaintenancePricePower: Hydraulic motor
            maintenance price [EUR/kW], or None.
    """

    winch: WinchCosts
    electricalStorage: StorageType
    ultracapacitor: StorageCosts
    battery: StorageCosts
    powerConverterPricePower: float
    drivetrain: Optional[DrivetrainType] = None
    gearbox: Optional[GearboxCosts] = None
    generator: Optional[GeneratorCosts] = None
    pumpMotorPricePower: Optional[float] = None
    pumpMotorMaintenancePricePower: Optional[float] = None
    hydraulicAccumulatorPriceEnergy: Optional[float] = None
    hydraulicAccumulatorMaintenancePriceEnergy: Optional[float] = None
    hydraulicMotorPricePower: Optional[float] = None
    hydraulicMotorMaintenancePricePower: Optional[float] = None

    def storage(self, storage_type: StorageType) -> StorageCosts:
        """Return the cost parameters of one storage type.

        Args:
            storage_type (StorageType): The storage type to look up.

        Returns:
            StorageCosts: Cost parameters of the storage type.
        """
        return (self.ultracapacitor
                if storage_type == StorageType.ULTRACAPACITOR
                else self.battery)


@dataclass(frozen=True)
class BalanceOfSystemCosts:
    """Balance of System subsystem cost parameters.

    Attributes:
        sitePreparationPricePower: Site preparation price [EUR/kW].
        foundationPricePower: Foundation price [EUR/kW].
        installationPricePower: Installation price [EUR/kW].
        operationsMaintenancePricePower: O&M price [EUR/kW/year].
        decommissioningInstallationFraction: Decommissioning cost as a
            fraction of the installation cost [-].
    """

    sitePreparationPricePower: float
    foundationPricePower: float
    installationPricePower: float
    operationsMaintenancePricePower: float
    decommissioningInstallationFraction: float


@dataclass(frozen=True)
class MarketCosts:
    """Market and electricity price parameters.

    Attributes:
        electricityPriceIntercept: Electricity price intercept
            [EUR/MWh].
        electricityPriceSlope: Electricity price slope
            [(EUR/MWh)/(m/s)].
        subsidy: Production subsidy [EUR/MWh].
    """

    electricityPriceIntercept: float
    electricityPriceSlope: float
    subsidy: float


@dataclass(frozen=True)
class EcoCosts:
    """Complete cost parameter set of one ECOMo simulation.

    Attributes:
        kite: Kite cost parameters.
        tether: Tether cost parameters.
        groundStation: Ground station cost parameters.
        balanceOfSystem: Balance of System cost parameters.
        market: Market and electricity price parameters.
    """

    kite: KiteCosts
    tether: TetherCosts
    groundStation: GroundStationCosts
    balanceOfSystem: BalanceOfSystemCosts
    market: MarketCosts

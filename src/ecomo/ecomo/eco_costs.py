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
        priceFabric: Soft-wing fabric price [EUR/m2] (flat-price model).
        priceBridle: Soft-wing bridle price [EUR/m2] (flat-price model).
        structureLifetime: Soft-wing structural lifetime at full
            loading [flying years]. Legacy calendar replacement model;
            used only when ``canopyLifetimeFlightHours`` is not set.
        canopyLifetimeFlightHours: Soft-wing canopy service life,
            expressed in loaded (reel-out) hours [h]. When set, the
            reel-out-hour replacement model is used: the canopy is
            consumed by accumulated reel-out (traction) hours rather
            than calendar time.
        perCyclePenalty: Optional canopy life fraction consumed per
            pumping cycle [-]; adds a cycle-count term to the reel-out-
            hour replacement model. As it multiplies the annual pumping-
            cycle count (hundreds of thousands per year), it is a tiny
            per-cycle fatigue fraction (~1e-6), not a per-deployment cost.
            None or 0 disables it.
        canopyLoadExponent: Optional S-N (Wohler) exponent m [-] for the
            load-weighted canopy life. When set (and non-zero), each
            reel-out hour is weighted by ``(F / F_ref)^m`` (Miner's rule
            with a power-law S-N curve), so partial-load hours consume
            less life. m = 0 or None reproduces the load-independent
            hour model. When set, ``canopyLifetimeFlightHours`` is the
            life at the reference load ``canopyReferenceForce``.
        canopyReferenceForce: Reference traction force F_ref [N] for the
            load-weighted canopy life; None uses the peak traction force
            in the power curves (i.e. the life is anchored at peak load).
        onboardGeneratorPricePower: Onboard generator price [EUR/kW],
            or None.
        onboardBatteryPriceEnergy: Onboard battery price [EUR/kWh], or
            None.
        materialCostRef: Soft-wing material cost at the reference area
            [EUR] (two-term model). When set, the two-term model is
            used instead of the flat-price model.
        referenceArea: Reference flat wing area S_ref [m2] (two-term
            model).
        materialScalingExponent: Material cost scaling exponent b_mat
            [-] (two-term model).
        labourCostCoefficient: Labour cost coefficient C_lab [EUR/m2]
            (two-term model).
        avionicsCostFixed: Fixed avionics/electronics cost [EUR] (area-
            scaled KCU model). When set, the scaled model is used
            instead of the flat avionicsCost.
        avionicsCostVarRef: Variable avionics/actuator cost at the
            reference area [EUR] (area-scaled KCU model).
        avionicsReferenceArea: Reference flat wing area for avionics
            scaling [m2].
        avionicsScalingExponent: Avionics cost scaling exponent [-].
        avionicsLifetime: Avionics/KCU service life [years]; drives a
            replacement OPEX ``(1/life) * CAPEX`` (capped at the project
            life). None means no avionics replacement is charged.
        sensorCost: Airborne sensor suite (GNSS/IMU) cost [EUR], itemised
            separately from the KCU electronics. None (treated as zero).
        sensorLifetime: Sensor service life [years]; drives a replacement
            OPEX ``(1/life) * CAPEX`` (capped at the project life). None
            means no sensor replacement is charged.
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
    canopyLifetimeFlightHours: Optional[float] = None
    perCyclePenalty: Optional[float] = None
    canopyLoadExponent: Optional[float] = None
    canopyReferenceForce: Optional[float] = None
    onboardGeneratorPricePower: Optional[float] = None
    onboardBatteryPriceEnergy: Optional[float] = None
    # Soft-wing two-term structure cost model (preferred over the flat
    # priceFabric/priceBridle model when set)
    materialCostRef: Optional[float] = None
    referenceArea: Optional[float] = None
    materialScalingExponent: Optional[float] = None
    labourCostCoefficient: Optional[float] = None
    # Area-scaled avionics (KCU) model (preferred over avionicsCost when set)
    avionicsCostFixed: Optional[float] = None
    avionicsCostVarRef: Optional[float] = None
    avionicsReferenceArea: Optional[float] = None
    avionicsScalingExponent: Optional[float] = None
    avionicsLifetime: Optional[float] = None
    # Airborne sensor suite, priced separately from the KCU electronics
    sensorCost: Optional[float] = None
    sensorLifetime: Optional[float] = None


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
            Legacy semi-log S-N intercept; used only when the master
            curve below is not configured (dormant fallback).
        bendingLifeA2: Bending fatigue coefficient a2 [-], or None.
        nBends: Number of bends (pulleys) per cycle [-], or None.
        operationalLife: Empirical operational (non-fatigue) tether life
            in flight hours [h], or None. Captures UV, abrasion, particle
            ingress and handling wear, which are not stress-driven. When
            set, it adds a degradation mode so the replacement frequency
            is governed by the shortest of bending, creep and operational
            life (fatigue still governs at high stress, the operational
            life governs at the low stress of soft-wing systems).
        masterCurveCoeff: Meuwissen/Bosman bearing-pressure master-curve
            coefficient C [-] (with p_N in MPa), or None. When set, the
            bending life uses ``N_f = C * p_N ** (-B)`` instead of the
            a1/a2 semi-log model.
        masterCurveExponent: Master-curve slope B [-], or None.
        bearingPressureCoeff: Bearing-pressure coefficient k_pw [-]
            (their Eq. 3, ``p_N = k_pw * sigma_MPa / (D/d)``), or None.
        pwLimitMpa: Low-pressure clamp on p_N [MPa], or None; bending
            actions below it are treated as occurring at it.
        bendingDdRatio: Winch drum-to-tether diameter ratio D/d [-] used
            by the master curve, or None. Taken from the ground station
            winch, threaded through the loader.
        designSafetyFactor: Bending retirement margin SF [-], or None.
            The replacement frequency is scaled up by SF (retire at
            CTF/SF, i.e. before cycles-to-failure).
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
    operationalLife: Optional[float] = None
    # Meuwissen/Bosman bearing-pressure bending master curve (preferred
    # over the a1/a2 semi-log model when configured)
    masterCurveCoeff: Optional[float] = None
    masterCurveExponent: Optional[float] = None
    bearingPressureCoeff: Optional[float] = None
    pwLimitMpa: Optional[float] = None
    bendingDdRatio: Optional[float] = None
    designSafetyFactor: Optional[float] = None


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
        lifetime: Drum/winch service life [years]; drives a replacement
            OPEX ``(1/life) * CAPEX`` when shorter than the project
            life. None means no replacement is charged.
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
    lifetime: Optional[float] = None

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
        lifetime: Gearbox service life [years]; drives a replacement
            OPEX ``(1/life) * CAPEX`` when shorter than the project
            life. None means no replacement is charged.
    """

    costModel: ComponentCostModel
    pricePower: Optional[float] = None
    priceMass: Optional[float] = None
    massCoefficient: Optional[float] = None
    massExponent: Optional[float] = None
    lifetime: Optional[float] = None


@dataclass(frozen=True)
class GeneratorCosts:
    """Electric generator cost parameters.

    Attributes:
        costModel: Generator cost model.
        pricePower: Power-based price [EUR/kW] (power-based model).
        priceMass: Mass-based price [EUR/kg] (mass-based model).
        massSlope: Mass scaling slope (mass-based model).
        massOffset: Mass offset (mass-based model).
        lifetime: Generator service life [years]; drives a replacement
            OPEX ``(1/life) * CAPEX`` when shorter than the project
            life. None means no replacement is charged.
    """

    costModel: ComponentCostModel
    pricePower: Optional[float] = None
    priceMass: Optional[float] = None
    massSlope: Optional[float] = None
    massOffset: Optional[float] = None
    lifetime: Optional[float] = None


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
        powerConverterLifetime: Power converter service life [years];
            drives a replacement OPEX ``(1/life) * CAPEX`` when shorter
            than the project life. None means no replacement is
            charged.
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
        launchLandCost: Launch & land (take-off & landing) system fixed
            cost [EUR], or None (treated as zero).
        launchLandPriceArea: Launch & land system area-specific price
            [EUR/m2]; when set, the L&L CAPEX is
            ``launchLandPriceArea * flat_wing_area`` and takes precedence
            over the fixed ``launchLandCost``. None uses the fixed cost.
        launchLandLifetime: Launch & land system service life [years],
            or None for no replacement.
    """

    winch: WinchCosts
    electricalStorage: StorageType
    ultracapacitor: StorageCosts
    battery: StorageCosts
    powerConverterPricePower: float
    powerConverterLifetime: Optional[float] = None
    drivetrain: Optional[DrivetrainType] = None
    gearbox: Optional[GearboxCosts] = None
    generator: Optional[GeneratorCosts] = None
    pumpMotorPricePower: Optional[float] = None
    pumpMotorMaintenancePricePower: Optional[float] = None
    hydraulicAccumulatorPriceEnergy: Optional[float] = None
    hydraulicAccumulatorMaintenancePriceEnergy: Optional[float] = None
    hydraulicMotorPricePower: Optional[float] = None
    hydraulicMotorMaintenancePricePower: Optional[float] = None
    launchLandCost: Optional[float] = None
    launchLandPriceArea: Optional[float] = None
    launchLandLifetime: Optional[float] = None

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
        operationsMaintenancePricePower: Per-kW O&M overhead price
            [EUR/kW/year] (size-dependent upkeep; no labour, no
            consumables).
        decommissioningInstallationFraction: Decommissioning cost as a
            fraction of the installation cost [-].
        consumablesEurPerYear: Recurring consumables bundle (AWT, kite
            bag, weak link, sensors) replaced yearly [EUR/year].
        consumablesMaturity: Maturity factor multiplying the consumables
            cost [-]; matures down (<1) as the system matures. Default 1.
    """

    sitePreparationPricePower: float
    foundationPricePower: float
    installationPricePower: float
    operationsMaintenancePricePower: float
    decommissioningInstallationFraction: float
    consumablesEurPerYear: float = 0.0
    consumablesMaturity: float = 1.0


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

"""Input structures for the ECOMo simulation.

The frozen dataclasses in this module hold the physical, financial and
performance inputs of one simulation. Derived quantities are resolved
when the inputs are built (e.g. the flat wing area) or exposed as
properties (e.g. the tether cross-sectional area).

Replacement frequencies use ``None`` to request auto-estimation from
the load and cycle models, ``0`` for components that are never
replaced, and a positive value [1/year] for a fixed replacement
schedule.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class Topology:
    """AWE system topology.

    Attributes:
        power: Power generation type, 'GG' (ground-gen) or 'FG'
            (fly-gen).
        wing: Wing type, 'fixed' or 'soft'.
    """

    power: str
    wing: str


@dataclass(frozen=True)
class BusinessInputs:
    """Financial parameters of the project.

    Attributes:
        nYears: Project lifetime [years].
        costOfDebt: Cost of debt [-].
        costOfEquity: Cost of equity [-].
        taxRate: Corporate tax rate [-].
        debtToEquity: Debt-to-equity ratio [-].
    """

    nYears: int
    costOfDebt: float
    costOfEquity: float
    taxRate: float
    debtToEquity: float

    @property
    def wacc(self) -> float:
        """Weighted average cost of capital (discount rate) [-]."""
        return (self.debtToEquity / (1 + self.debtToEquity) *
                self.costOfDebt * (1 - self.taxRate) +
                1 / (1 + self.debtToEquity) * self.costOfEquity)


@dataclass(frozen=True)
class KiteInputs:
    """Physical kite parameters.

    Attributes:
        mass: Structural wing mass [kg].
        flatArea: Flat wing area used by the cost model [m2].
        structureReplacementFrequency: Structure replacements per year
            [1/year]; None requests auto-estimation (soft wings only).
        onboardGeneratorPower: Onboard generator rated power [W]
            (GG systems), or None.
        onboardBatteryCapacity: Onboard battery capacity [kWh]
            (GG systems), or None.
        uniaxMass: Uniax laminate mass [kg] for the laminate cost
            model, or None.
        triaxMass: Triax laminate mass [kg] for the laminate cost
            model, or None.
    """

    mass: float
    flatArea: float
    structureReplacementFrequency: Optional[float] = None
    onboardGeneratorPower: Optional[float] = None
    onboardBatteryCapacity: Optional[float] = None
    uniaxMass: Optional[float] = None
    triaxMass: Optional[float] = None


@dataclass(frozen=True)
class TetherInputs:
    """Physical tether parameters.

    Attributes:
        diameter: Worked-in tether diameter [m].
        length: Tether length [m].
        density: Fibre density [kg/m3].
        replacementFrequency: Tether replacements per year [1/year];
            None requests auto-estimation from the bending and creep
            life models.
    """

    diameter: float
    length: float
    density: float
    replacementFrequency: Optional[float] = None

    @property
    def area(self) -> float:
        """Tether cross-sectional area [m2]."""
        return np.pi / 4 * self.diameter ** 2


@dataclass(frozen=True)
class StorageInputs:
    """Parameters of one energy storage bank.

    Attributes:
        ratedCapacity: Rated storage capacity [kWh].
        exchangedEnergy: Energy exchanged per pumping cycle [kWh],
            scalar or per wind speed; None when not applicable.
        replacementFrequency: Replacements per year [1/year]; None
            requests auto-estimation from the cycle count.
    """

    ratedCapacity: float
    exchangedEnergy: Optional[Union[float, np.ndarray]] = None
    replacementFrequency: Optional[float] = None


@dataclass(frozen=True)
class GroundStationInputs:
    """Ground station storage and maintenance parameters.

    Attributes:
        ultracapacitor: Ultracapacitor bank, or None if absent.
        battery: Battery bank, or None if absent.
        hydraulicAccumulator: Hydropneumatic accumulator bank, or None
            if absent.
        hydraulicMotorReplacementFrequency: Hydraulic motor major
            maintenance frequency [1/year], or None.
        pumpMotorReplacementFrequency: Pump-motor major maintenance
            frequency [1/year], or None.
    """

    ultracapacitor: Optional[StorageInputs] = None
    battery: Optional[StorageInputs] = None
    hydraulicAccumulator: Optional[StorageInputs] = None
    hydraulicMotorReplacementFrequency: Optional[float] = None
    pumpMotorReplacementFrequency: Optional[float] = None


@dataclass(frozen=True)
class PerformanceData:
    """Wind-dependent system performance data.

    Populated from the performance file in standalone mode or from the
    AWESPA output files in AWESPA-connected mode.

    Attributes:
        windSpeeds: Wind speeds [m/s].
        windPdf: Wind speed probability density at each wind speed.
        averagePower: Average electrical cycle power [W] per wind
            speed.
        ratedPower: Rated electrical power [W].
        tetherForce: Tether force [N] per wind speed.
        peakMechanicalPower: Peak mechanical reel-out power [W]
            (GG systems), or None.
        cycleTime: Pumping cycle duration [s], scalar or per wind
            speed; None when not applicable.
        tipSpeedRatio: Wing speed to wind speed ratio [-] (FG
            systems), or None.
        turningRadius: Loop turning radius [m] (FG systems), or None.
        externalAep: Externally computed annual energy production
            [MWh] (AWESPA-connected mode), or None to integrate the
            power curve over the wind distribution.
        reelOutTimeFraction: Reel-out time fraction t_reel_out /
            t_cycle per wind speed [-], used by the time-weighted
            soft-wing loading factor; None for FG systems or when the
            timing data are unavailable (falls back to the unweighted
            loading factor with a warning).
    """

    windSpeeds: np.ndarray
    windPdf: np.ndarray
    averagePower: np.ndarray
    ratedPower: float
    tetherForce: np.ndarray
    peakMechanicalPower: Optional[float] = None
    cycleTime: Optional[Union[float, np.ndarray]] = None
    tipSpeedRatio: Optional[float] = None
    turningRadius: Optional[float] = None
    externalAep: Optional[float] = None
    reelOutTimeFraction: Optional[np.ndarray] = None


@dataclass(frozen=True)
class EcoInputs:
    """Complete input set of one ECOMo simulation.

    Attributes:
        topology: System topology.
        business: Financial parameters.
        kite: Kite parameters.
        tether: Tether parameters.
        groundStation: Ground station parameters.
        performance: System performance data.
    """

    topology: Topology
    business: BusinessInputs
    kite: KiteInputs
    tether: TetherInputs
    groundStation: GroundStationInputs
    performance: PerformanceData

"""ECOMo: Reference Economic Model for Airborne Wind Energy Systems.

Python implementation of the AWE-Eco model originally developed in
MATLAB.
"""

from .base import EconomicModel
from .ecomo_economic import EcoMoEconomicModel

__version__ = "1.0.0"

__all__ = ["EconomicModel", "EcoMoEconomicModel"]

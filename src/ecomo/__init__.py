"""ECOMo: Reference Economic Model for Airborne Wind Energy Systems.

Python implementation of the AWE-Eco model originally developed in
MATLAB.
"""

from .base import EconomicModel
from .ecomo_economic import EcoMo

__version__ = "1.0.0"

__all__ = ["EconomicModel", "EcoMo"]

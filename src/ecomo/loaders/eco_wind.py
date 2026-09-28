"""Wind resource helpers for the ECOMo simulation."""

import numpy as np

# Default Weibull wind distribution parameters (onshore reference case)
WEIBULL_SHAPE = 2
WEIBULL_SCALE = 8


def weibull_pdf(windRange: np.ndarray,
                shape: float = WEIBULL_SHAPE,
                scale: float = WEIBULL_SCALE) -> np.ndarray:
    """Evaluate the Weibull wind speed probability density.

    Args:
        windRange (np.ndarray): Wind speeds [m/s].
        shape (float): Weibull shape parameter. Defaults to
            WEIBULL_SHAPE.
        scale (float): Weibull scale parameter. Defaults to
            WEIBULL_SCALE.

    Returns:
        np.ndarray: Probability density at each wind speed.
    """
    return (shape / scale * (windRange / scale) ** (shape - 1) *
            np.exp(-(windRange / scale) ** shape))

"""Loader for the site wind resource.

Reduces an awesIO ``wind_resource_schema.yml`` file to the marginal
wind-speed probability density used by the economic integrals, summed
over wind profiles and directions.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

try:
    from awesio.validator import validate as awesio_validate  # type: ignore
except ImportError:
    awesio_validate = None


def eco_load_wind_resource(wind_resource_path: Path,
                           validate: bool = True
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Load the site wind-speed probability density from a YAML file.

    Reads an awesIO wind resource file and reduces its probability
    matrix (profiles x wind speeds x directions) to the marginal
    wind-speed distribution, expressed as a probability density
    [1/(m/s)] at each wind-speed bin center.

    Args:
        wind_resource_path (Path): Path to a wind resource YAML file
            (awesIO ``wind_resource_schema.yml`` format).
        validate (bool): If True, attempt awesIO validation of the
            file. Defaults to True.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the wind-speed bin centers are missing or do not
            match the probability matrix.

    Returns:
        tuple: ``(windSpeeds, density)`` with the bin-center wind speeds
        [m/s] and the probability density at each [1/(m/s)].
    """
    resourcePath = Path(wind_resource_path)
    if not resourcePath.exists():
        raise FileNotFoundError(
            f"Wind resource file not found: {resourcePath}")

    with open(resourcePath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if validate and awesio_validate is not None:
        try:
            awesio_validate(data)
        except Exception as e:
            print(f"Note: awesIO validation skipped for "
                  f"{resourcePath.name}: {e}")

    bins = data.get('wind_speed_bins') or {}
    binCenters = bins.get('bin_centers')
    if binCenters is None:
        raise ValueError(
            f"Missing 'wind_speed_bins.bin_centers' in {resourcePath}")
    binCenters = np.asarray(binCenters, dtype=float)

    # Probability matrix shape: [n_profiles, n_wind_speeds, n_directions]
    matrix = np.asarray(data['probability_matrix']['data'], dtype=float)
    binMass = matrix.sum(axis=(0, 2))
    if binMass.shape != binCenters.shape:
        raise ValueError(
            f"The probability matrix has {binMass.shape[0]} wind speed "
            f"bins but 'bin_centers' has {binCenters.shape[0]} in "
            f"{resourcePath}")

    # Convert the per-bin probability mass to a probability density
    binEdges = bins.get('bin_edges')
    if binEdges is not None:
        binWidths = np.diff(np.asarray(binEdges, dtype=float))
    else:
        binWidths = np.gradient(binCenters)

    probability = binMass / binMass.sum()
    density = probability / binWidths

    return binCenters, density

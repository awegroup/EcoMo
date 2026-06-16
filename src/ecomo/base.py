"""Abstract base class for economic models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class EconomicModel(ABC):
    """Abstract base class for AWE economic models.

    All economic models must inherit from this class and implement the
    required methods for loading configuration and computing economic
    metrics.
    """

    @abstractmethod
    def load_configuration(
        self,
        economic_settings_path: Path,
        validate: bool = True,
    ) -> None:
        """Load economic model configuration from YAML files.

        Args:
            economic_settings_path (Path): Path to the economic
                settings YAML file (awesIO format). The settings file
                references the system, cost and performance input
                files it needs.
            validate (bool): If True, validate configuration files
                using the awesIO validator when a matching schema is
                available. Defaults to True.
        """
        pass

    @abstractmethod
    def compute_economics(
        self,
        output_path: Path = None,
        verbose: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Compute economic metrics and optionally export results.

        This method runs the economic model, exports the results to
        YAML if ``output_path`` is provided, and returns the results
        structure.

        Args:
            output_path (Path): Path where the results YAML will be
                written. If None, no export is performed.
            verbose (bool): Whether to print a summary of the computed
                metrics. Defaults to False.
            validate (bool): If True, validate the output YAML file
                using the awesIO validator when a matching schema is
                available. Defaults to True.

        Returns:
            dict: Economic results and metrics.
        """
        pass

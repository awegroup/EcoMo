"""Configuration settings for ECOMo simulation.

All settings are controlled from the entry point (run_ecomo.py) via
the configure() function.  This module only holds the shared state.
"""

from dataclasses import dataclass


@dataclass
class Settings:
    """Configuration settings for ECOMo simulation.

    Attributes:
        name: Name identifier for the output files.
        input_cost_file: Name of the Excel file containing cost inputs.
        input_model_file: Name of the input model file or 'code' for
            programmatic inputs.
        power: Power generation type ('FG' for fly-gen or 'GG' for
            ground-gen).
        wing: Wing type ('fixed' or 'soft').
    """

    name: str = ''
    input_cost_file: str = ''
    input_model_file: str = ''
    power: str = ''
    wing: str = ''


# Shared global settings instance
eco_settings = Settings()


def configure(**kwargs) -> None:
    """Update global settings from keyword arguments.

    Args:
        **kwargs: Keyword arguments matching Settings fields
            (name, input_cost_file, input_model_file, power, wing).
    """
    for key, value in kwargs.items():
        if hasattr(eco_settings, key):
            setattr(eco_settings, key, value)
        else:
            raise ValueError(f"Unknown setting: {key}")

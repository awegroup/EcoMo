"""Import model parameters from an Excel file.

This module reads model parameters from an Excel file and organizes them
into a structured format for use in the ECOMo simulation.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from .config import eco_settings


def _parse_excel_value(value):
    """Convert an Excel cell value to a Python/NumPy type.

    Handles bracket-delimited arrays (e.g. ``"[4 5 6 7 8]"``)
    and plain number strings. Non-numeric strings are returned as-is.

    Args:
        value: Raw cell value from pandas read_excel.

    Returns:
        float, numpy array, or original value.
    """
    if not isinstance(value, str):
        return value
    s = value.strip().rstrip("'")
    if s.startswith('[') and s.endswith(']'):
        parts = s[1:-1].replace(',', ' ').split()
        try:
            return np.array([float(x) for x in parts])
        except ValueError:
            return value
    try:
        return float(s)
    except (ValueError, TypeError):
        return value


def eco_import_model(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Import model parameters from an Excel file.

    Reads model parameters from an Excel file specified in eco_settings
    and merges them into the existing input dictionary.

    Args:
        inp: Dictionary containing existing input parameters.

    Returns:
        Updated dictionary containing imported model parameters.
    """
    # Sheets to read (matching MATLAB eco_import_model.m)
    sheets = ['atm', 'kite', 'tether', 'system', 'gStation']

    # Base path for input data
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data'
    )
    file_path = os.path.join(base_path, f'{eco_settings.input_model_file}.xlsx')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model input file not found: {file_path}")

    for sheet in sheets:
        try:
            df = pd.read_excel(
                file_path, sheet_name=sheet, header=None, nrows=30
            )

            # Ensure sheet key exists in inp
            if sheet not in inp:
                inp[sheet] = {}

            for idx, row in df.iterrows():
                param_name = row.iloc[0]
                param_value = row.iloc[1]

                # Skip missing rows
                if pd.isna(param_name):
                    continue

                param_name = str(param_name).strip()

                # Parse the value (handles MATLAB arrays, numbers, etc.)
                param_value = _parse_excel_value(param_value)

                # Handle nested structure via dot notation
                parts = param_name.split('.')

                if len(parts) == 1:
                    inp[sheet][parts[0]] = param_value
                elif len(parts) == 2:
                    if parts[0] not in inp[sheet]:
                        inp[sheet][parts[0]] = {}
                    inp[sheet][parts[0]][parts[1]] = param_value

        except Exception as e:
            print(f"Warning: Could not read sheet '{sheet}' from {file_path}: {e}")

    return inp

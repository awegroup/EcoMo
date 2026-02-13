"""Import cost parameters from Excel files.

This module reads cost parameters from Excel files and organizes them
into a nested dictionary structure for use in the ECOMo simulation.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any
from .config import eco_settings


def _parse_excel_value(value):
    """Convert an Excel cell value to a Python/NumPy type.

    Handles bracket-delimited arrays (e.g. ``"[-2.4 8.3 -11.2 5.2]'"``)
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


def eco_import_cost_par() -> Dict[str, Any]:
    """Import cost parameters from an Excel file.

    Reads cost parameters from an Excel file specified in eco_settings
    and organizes them into a nested dictionary structure.

    Returns:
        Dictionary containing imported cost parameters organized by sheet.
    """
    par = {}

    # Define sheets to read
    sheets = ['kite', 'tether', 'gStation', 'BoS', 'BoP', 'metrics']

    # Base path for input data
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data'
    )
    file_path = os.path.join(base_path, f'{eco_settings.input_cost_file}.xlsx')

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cost input file not found: {file_path}")

    # Read each sheet
    for sheet in sheets:
        try:
            # Read the sheet with first column as parameter name, second as value
            df = pd.read_excel(file_path, sheet_name=sheet, header=None, nrows=50)

            # Initialize sheet dictionary
            par[sheet] = {}

            # Process each row
            for idx, row in df.iterrows():
                param_name = row.iloc[0]
                param_value = row.iloc[1]

                # Skip if parameter name is missing
                if pd.isna(param_name):
                    continue

                param_name = str(param_name)

                # Parse MATLAB-style values (string arrays, numbers, etc.)
                param_value = _parse_excel_value(param_value)

                # Handle nested structures based on dots in parameter name
                parts = param_name.split('.')

                if len(parts) == 1:
                    # Top-level parameter
                    par[sheet][parts[0]] = param_value
                elif len(parts) == 2:
                    # One level deep
                    if parts[0] not in par[sheet]:
                        par[sheet][parts[0]] = {}
                    par[sheet][parts[0]][parts[1]] = param_value
                elif len(parts) == 3:
                    # Two levels deep
                    if parts[0] not in par[sheet]:
                        par[sheet][parts[0]] = {}
                    if parts[1] not in par[sheet][parts[0]]:
                        par[sheet][parts[0]][parts[1]] = {}
                    par[sheet][parts[0]][parts[1]][parts[2]] = param_value
                elif len(parts) == 4:
                    # Three levels deep
                    if parts[0] not in par[sheet]:
                        par[sheet][parts[0]] = {}
                    if parts[1] not in par[sheet][parts[0]]:
                        par[sheet][parts[0]][parts[1]] = {}
                    if parts[2] not in par[sheet][parts[0]][parts[1]]:
                        par[sheet][parts[0]][parts[1]][parts[2]] = {}
                    par[sheet][parts[0]][parts[1]][parts[2]][parts[3]] = param_value

        except Exception as e:
            print(f"Warning: Could not read sheet '{sheet}' from {file_path}: {e}")
            par[sheet] = {}

    return par

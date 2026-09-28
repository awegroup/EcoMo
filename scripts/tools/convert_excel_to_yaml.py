# NOTE: Development artifact, run once to convert the original Excel input files
# to YAML. The Excel files in data/ and this script can be archived once the
# generated YAML files in config/test/ are verified. This script is not intended to be included
# in the normal model workflow.
"""Convert the legacy ECOMo Excel input files to YAML input files.

One-off migration script. Reads the six Excel files in ``data/`` and
writes, per topology case, three YAML files into ``config/example/``:

- ``economic_cost_inputs_<case>.yml``  (cost model parameters)
- ``system_<case>.yml``                (awesIO system_schema.yml format)
- ``system_performance_<case>.yml``    (performance-model outputs)

The Excel values are carried over verbatim; only key names, type codes
(integer -> string) and units (storage kWh -> Wh in the system YAML)
are translated. Data oddities are noted in the metadata of the
generated files rather than silently corrected.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "config" / "example"

AWESIO_VERSION = "0.1.0"
KWH_TO_WH = 1e3

CASES = {
    "GG_fixed": {
        "cost_file": "eco_cost_inputs_GG_fixed",
        "system_file": "eco_system_inputs_GG_fixed_example",
        "power": "GG",
        "wing": "fixed",
    },
    "GG_soft": {
        "cost_file": "eco_cost_inputs_GG_soft",
        "system_file": "eco_system_inputs_GG_soft_example",
        "power": "GG",
        "wing": "soft",
    },
    "FG": {
        "cost_file": "eco_cost_inputs_FG",
        "system_file": "eco_system_inputs_FG_example",
        "power": "FG",
        "wing": "fixed",
    },
}

WINCH_MATERIAL_CODES = {1: "aluminum", 2: "steel"}
DRIVETRAIN_CODES = {1: "electric", 2: "hydraulic"}
STORAGE_CODES = {1: "ultracapacitor", 2: "battery"}
COST_MODEL_CODES = {1: "power_based", 2: "mass_based"}
KITE_COST_MODEL_CODES = {1: "mass_area", 2: "laminate"}


# --------------------------------------------------------------------- #
#  Excel reading (same parsing rules as the legacy importers)
# --------------------------------------------------------------------- #

def parse_excel_value(value):
    """Convert an Excel cell value to a Python/NumPy type."""
    if not isinstance(value, str):
        return value
    stripped = value.strip().rstrip("'")
    if stripped.startswith("[") and stripped.endswith("]"):
        parts = stripped[1:-1].replace(",", " ").split()
        try:
            return np.array([float(x) for x in parts])
        except ValueError:
            return value
    try:
        return float(stripped)
    except (ValueError, TypeError):
        return value


def read_excel_sheets(file_path, sheets):
    """Read dotted-name parameter sheets into a nested dict."""
    data = {}
    for sheet in sheets:
        data[sheet] = {}
        df = pd.read_excel(file_path, sheet_name=sheet, header=None)
        for _, row in df.iterrows():
            name = row.iloc[0]
            if pd.isna(name):
                continue
            value = parse_excel_value(row.iloc[1])
            node = data[sheet]
            parts = str(name).strip().split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
    return data


def to_builtin(value):
    """Convert numpy types to plain Python types for YAML output."""
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_builtin(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_yaml(path, data, header_comment=None):
    """Write a YAML file, optionally with a leading comment block."""
    text = yaml.safe_dump(to_builtin(data), sort_keys=False,
                          default_flow_style=False, width=100)
    if header_comment:
        comment = "\n".join(f"# {line}" for line in header_comment)
        text = comment + "\n" + text
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


def metadata(name, description, note, schema):
    return {
        "name": name,
        "description": description,
        "note": note,
        "awesIO_version": AWESIO_VERSION,
        "schema": schema,
    }


# --------------------------------------------------------------------- #
#  Cost inputs conversion
# --------------------------------------------------------------------- #

def convert_cost_inputs(case, info):
    """Convert one cost Excel file to economic_cost_inputs_<case>.yml."""
    raw = read_excel_sheets(DATA_DIR / f"{info['cost_file']}.xlsx",
                            ["kite", "tether", "gStation", "BoS", "BoP",
                             "metrics"])

    # --- kite ---------------------------------------------------------
    kiteRaw = raw["kite"]
    kite = {"structure": {}}
    if "fixed" in kiteRaw.get("structure", {}):
        fixedRaw = kiteRaw["structure"]["fixed"]
        kite["structure"]["fixed"] = {
            "cost_model": KITE_COST_MODEL_CODES[int(fixedRaw["approach"])],
            "mass_area": {
                "price_mass": fixedRaw["one"]["p_str"],          # EUR/kg
                "price_wetted_area": fixedRaw["one"]["p_wet"],   # EUR/m2
            },
            "laminate": {
                "price_uniax": fixedRaw["two"]["p_uni"],         # EUR/kg
                "price_triax": fixedRaw["two"]["p_tri"],         # EUR/kg
                "manufacturing_factor": fixedRaw["two"]["f_man"],
            },
        }
    if "soft" in kiteRaw.get("structure", {}):
        softRaw = kiteRaw["structure"]["soft"]
        kite["structure"]["soft"] = {
            "price_fabric": softRaw["p_fabric"],                 # EUR/m2
            "price_bridle": softRaw["p_bridle"],                 # EUR/m2
            "lifetime_flying_years": softRaw["L_str"],           # years
        }
    if "obGen" in kiteRaw:
        kite["onboard_generator"] = {"price_power": kiteRaw["obGen"]["p"]}
    if "obBatt" in kiteRaw:
        kite["onboard_battery"] = {"price_energy": kiteRaw["obBatt"]["p"]}
    kite["avionics"] = {"cost": kiteRaw["avio"]["C"]}

    # --- tether -------------------------------------------------------
    tetherRaw = raw["tether"]
    tether = {
        "price_mass": tetherRaw["p"],                            # EUR/kg
        "conductive_manufacturing_factor": tetherRaw["f_mt"],
        "creep_life_coefficients": tetherRaw["L_creep"],
        "coating_mass_fraction": tetherRaw["f_coat"],
        "fibre_area_fraction": tetherRaw["f_At"],
        "bending_life_a1": tetherRaw["a_1b"],
        "bending_life_a2": tetherRaw["a_2b"],
        "n_bends": tetherRaw["N_bends"],
        "max_stress": tetherRaw["sigma_max"],                    # Pa
    }

    # --- ground station -----------------------------------------------
    gsRaw = raw["gStation"]
    winchRaw = gsRaw["winch"]
    groundStation = {
        "winch": {
            "material": WINCH_MATERIAL_CODES[int(winchRaw["material"])],
            "drum_to_tether_diameter_ratio": winchRaw["dwinch_dt"],
            "safety_factor_diameter": winchRaw["SF_dt"],
            "safety_factor_length": winchRaw["SF_Lt"],
            "materials": {
                "aluminum": {
                    "price_mass": winchRaw["p_al"],              # EUR/kg
                    "density": winchRaw["rho_al"],               # kg/m3
                    "max_stress": winchRaw["sigma_al"],          # Pa
                },
                "steel": {
                    "price_mass": winchRaw["p_st"],
                    "density": winchRaw["rho_st"],
                    "max_stress": winchRaw["sigma_st"],
                },
            },
        },
    }
    if "drivetrain_type" in gsRaw:
        groundStation["drivetrain"] = DRIVETRAIN_CODES[
            int(gsRaw["drivetrain_type"])]
    if "gearbox" in gsRaw:
        groundStation["gearbox"] = {
            "cost_model": COST_MODEL_CODES[int(gsRaw["gearbox"]["approach"])],
            "power_based": {"price_power": gsRaw["gearbox"]["one"]["p"]},
            "mass_based": {
                "price_mass": gsRaw["gearbox"]["two"]["p"],
                "mass_coefficient": gsRaw["gearbox"]["two"]["k"],
                "mass_exponent": gsRaw["gearbox"]["two"]["b"],
            },
        }
    if "gen" in gsRaw:
        groundStation["generator"] = {
            "cost_model": COST_MODEL_CODES[int(gsRaw["gen"]["approach"])],
            "power_based": {"price_power": gsRaw["gen"]["one"]["p"]},
            "mass_based": {
                "price_mass": gsRaw["gen"]["two"]["p"],
                "mass_slope": gsRaw["gen"]["two"]["k"],
                "mass_offset": gsRaw["gen"]["two"]["b"],
            },
        }
    groundStation["electrical_storage"] = STORAGE_CODES[
        int(gsRaw["elecSto_type"])]
    groundStation["ultracapacitor"] = {
        "price_energy": gsRaw["ultracap"]["p"],                  # EUR/kWh
        "cycle_life": gsRaw["ultracap"]["N"],
    }
    groundStation["battery"] = {
        "price_energy": gsRaw["batt"]["p"],
        "cycle_life": gsRaw["batt"]["N"],
    }
    groundStation["power_converter"] = {
        "price_power": gsRaw["powerConv"]["p"]}                  # EUR/kW
    if "pumpMotor" in gsRaw:
        groundStation["pump_motor"] = {
            "price_power": gsRaw["pumpMotor"]["p_1"],
            "maintenance_price_power": gsRaw["pumpMotor"]["p_2"],
        }
    if "hydAccum" in gsRaw:
        groundStation["hydraulic_accumulator"] = {
            "price_energy": gsRaw["hydAccum"]["p_1"],
            "maintenance_price_energy": gsRaw["hydAccum"]["p_2"],
        }
    if "hydMotor" in gsRaw:
        groundStation["hydraulic_motor"] = {
            "price_power": gsRaw["hydMotor"]["p_1"],
            "maintenance_price_power": gsRaw["hydMotor"]["p_2"],
        }

    # --- BoS / BoP / market ---------------------------------------------
    bosRaw = raw["BoS"]
    balanceOfSystem = {
        "site_preparation": {"price_power": bosRaw["sitePrep"]["p"]},
        "foundation": {"price_power": bosRaw["found"]["p"]},
        "installation": {"price_power": bosRaw["install"]["p"]},
        "operations_maintenance": {"price_power": bosRaw["OM"]["p"]},
        "decommissioning": {"installation_fraction": bosRaw["decomm"]["f"]},
    }

    data = {
        "metadata": metadata(
            f"ECOMo {case.replace('_', ' ')} Cost Inputs",
            "Economic cost model parameters for the ECOMo economic model",
            f"Converted from {info['cost_file']}.xlsx. Prices in EUR; "
            "power-specific prices in EUR/kW, energy-specific in EUR/kWh, "
            "mass-specific in EUR/kg, area-specific in EUR/m2.",
            "economic_schema.yml",
        ),
        "costs": {
            "kite": kite,
            "tether": tether,
            "ground_station": groundStation,
            "balance_of_system": balanceOfSystem,
            "balance_of_plant": raw["BoP"] or {},
        },
        "market": {
            "electricity_price": {
                "intercept": raw["metrics"]["electricity"]["p_0"],
                "wind_speed_slope": raw["metrics"]["electricity"]["p_1"],
            },
            "subsidy": raw["metrics"]["subsidy"],
        },
    }
    write_yaml(OUTPUT_DIR / f"economic_cost_inputs_{case}.yml", data)


# --------------------------------------------------------------------- #
#  System + performance conversion
# --------------------------------------------------------------------- #

def convert_system_and_performance(case, info):
    """Split one system Excel file into system + performance YAML."""
    raw = read_excel_sheets(DATA_DIR / f"{info['system_file']}.xlsx",
                            ["atm", "kite", "tether", "system", "gStation"])
    kiteRaw = raw["kite"]["structure"]
    tetherRaw = raw["tether"]
    systemRaw = raw["system"]
    gsRaw = raw["gStation"]
    isSoft = info["wing"] == "soft"
    isFlyGen = info["power"] == "FG"

    # ----- awesIO system YAML -------------------------------------------
    notes = [f"Converted from {info['system_file']}.xlsx."]
    wingStructure = {"mass": kiteRaw["m"]}
    if "A" in kiteRaw:
        # The Excel wing area is the area used directly by the cost
        # model (flat area). projected_surface_area is required by the
        # schema; the cost model reads flat_wing_area when present.
        wingStructure["projected_surface_area"] = kiteRaw["A"]
        wingStructure["flat_wing_area"] = kiteRaw["A"]
        notes.append("flat_wing_area carries the Excel wing area "
                     "verbatim; projected_surface_area is set to the "
                     "same value for schema compliance.")
    if "b" in kiteRaw:
        wingStructure["span"] = kiteRaw["b"]
    if "AR" in kiteRaw:
        wingStructure["aspect_ratio"] = kiteRaw["AR"]
    if "b" in kiteRaw and "A" not in kiteRaw:
        wingStructure["projected_surface_area"] = (
            kiteRaw["b"] ** 2 / kiteRaw["AR"])

    kite = {
        "name": f"ECOMo {case} example kite",
        "type": "soft_kite_kcu" if isSoft else "fixed_wing_gc",
        "version": 1.0,
        "joints": {"kite_joint": {"id": "K1"}},
        "wing": {
            "name": f"ECOMo {case} example wing",
            "type": "LEI_soft_kite" if isSoft else "fixed_wing",
            "version": 1.0,
            "structure": wingStructure,
        },
    }

    tether = {
        "name": "tether",
        "type": "conductive_tether" if isFlyGen else "non_conductive_tether",
        "version": 1.0,
        "joints": {
            "upper_tether_joint": {"id": "T1"},
            "lower_tether_joint": {"id": "T2"},
        },
        "structure": {
            "length": tetherRaw["L"],
            "diameter": tetherRaw["d"],
            "density": tetherRaw["rho"],
            "conductive": isFlyGen,
            "material": {
                "type": "dyneema",
                "breaking_strength": 1.5e9,  # Pa, DM20 fibre strength
            },
        },
    }

    storages = []
    if "ultracap" in gsRaw:
        storages.append({
            "name": "ultracapacitor bank",
            "type": "capacitor_bank",
            "capacity": gsRaw["ultracap"]["E_rated"] * KWH_TO_WH,  # Wh
        })
    if "batt" in gsRaw:
        battCapacity = gsRaw["batt"]["E_rated"] * KWH_TO_WH
        storages.append({
            "name": "battery bank",
            "type": "battery_bank",
            "capacity": battCapacity,                              # Wh
        })
        if case == "FG":
            notes.append("battery_bank capacity carried verbatim from "
                         "Excel (E_rated=100000); the unit looks "
                         "inconsistent (likely Wh, not kWh) but the "
                         "battery is unused with ultracapacitor storage.")

    systemData = {
        "metadata": metadata(
            f"ECOMo {case.replace('_', ' ')} Example System",
            f"Physical system parameters of the ECOMo {case} example case",
            " ".join(notes),
            "system_schema.yml",
        ),
        "assembly": {"connectivity_matrix": [["K1", "T1"], ["T2", "G1"]]},
        "components": {
            "kites": [kite],
            "tethers": [tether],
            "ground_station": {
                "name": f"ECOMo {case} example ground station",
                "type": ("fly_gen_station" if isFlyGen
                         else "pumping_ground_gen_station"),
                "version": 1.0,
                "joints": {"drum_joint": {"id": "G1"}},
                "structure": {"mass": 0.0},
                "storages": storages,
            },
        },
    }
    write_yaml(OUTPUT_DIR / f"system_{case}.yml", systemData)

    # ----- performance YAML ----------------------------------------------
    perfNotes = [f"Converted from {info['system_file']}.xlsx.",
                 "Field names aligned with power_curves_schema.yml "
                 "where possible."]
    performance = {
        "reference_wind_speeds": raw["atm"]["wind_range"],         # m/s
        "rated_electrical_power": systemRaw["P_e_rated"],          # W
    }
    if "P_m_peak" in systemRaw:
        performance["peak_mechanical_power"] = systemRaw["P_m_peak"]  # W
    performance["average_cycle_power"] = systemRaw["P_e_avg"]      # W
    if "Dt_cycle" in systemRaw:
        performance["cycle_time"] = systemRaw["Dt_cycle"]          # s
    performance["tether_force"] = systemRaw["F_t"]                 # N
    if "lambda" in systemRaw:
        performance["tip_speed_ratio"] = systemRaw["lambda"]
    if "R0" in systemRaw:
        performance["turning_radius"] = systemRaw["R0"]            # m

    exchangedEnergy = {}
    componentNames = {"ultracap": "ultracapacitor", "batt": "battery",
                      "hydAccum": "hydraulic_accumulator"}
    for excelKey, yamlKey in componentNames.items():
        if excelKey in gsRaw and "E_ex" in gsRaw[excelKey]:
            exchangedEnergy[yamlKey] = gsRaw[excelKey]["E_ex"]     # kWh
    if exchangedEnergy:
        performance["storage_exchanged_energy"] = exchangedEnergy
        if case == "GG_soft":
            perfNotes.append("storage_exchanged_energy.battery carried "
                             "verbatim from Excel (148930); the unit "
                             "looks inconsistent with the kWh values of "
                             "the other components.")

    perfData = {"metadata": metadata(
        f"ECOMo {case.replace('_', ' ')} Reference Performance",
        "Performance-model outputs used by the economic model in "
        "standalone mode",
        " ".join(perfNotes),
        "system_performance_schema.yml",
    )}
    perfData.update(performance)
    write_yaml(OUTPUT_DIR / f"system_performance_{case}.yml", perfData)


def main():
    """Convert all Excel inputs to YAML."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for case, info in CASES.items():
        print(f"Converting case {case}...")
        convert_cost_inputs(case, info)
        convert_system_and_performance(case, info)
    print("Done.")


if __name__ == "__main__":
    main()

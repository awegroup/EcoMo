"""ECOMo economic model for airborne wind energy systems.

The model is configured from an awesIO YAML settings file that
references the input files it needs. It can run in two modes,
selected by the ``input_files`` section of the settings:

- Standalone mode (default): system performance data come from the
  performance YAML.
- AWESPA-connected mode: when ``aep_results`` and ``power_curves`` are
  both set, AEP and system performance data are taken from those
  AWESPA output files instead of the performance YAML.
"""

import datetime
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import yaml

from .base import EconomicModel
from .ecomo.eco_main import eco_main
from .ecomo.eco_costs import EcoCosts
from .ecomo.eco_inputs import (
    BusinessInputs,
    EcoInputs,
    GroundStationInputs,
    KiteInputs,
    PerformanceData,
    StorageInputs,
    TetherInputs,
    Topology,
)
from .ecomo.loaders import (
    eco_load_cost_inputs,
    eco_load_system,
    eco_load_performance,
    peak_mechanical_power_fallback,
    eco_load_wind_resource,
    weibull_pdf,
    WEIBULL_SHAPE,
    WEIBULL_SCALE,
)
from .ecomo.eco_display_results import eco_display_results

try:
    from awesio.validator import validate as awesio_validate  # type: ignore
except ImportError:
    awesio_validate = None

AWESIO_VERSION = "0.1.0"
VALID_POWER_TYPES = ('GG', 'FG')
VALID_WING_TYPES = ('fixed', 'soft')

# Turning radius to wingspan ratio (Joshi & Trevisi 2024, Eq. 23)
TURNING_RADIUS_TO_SPAN = 5


def _parse_power_curves(power_curves_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract system performance arrays from awesIO power curve data.

    Handles both single-profile files (e.g. from the Luchsinger model)
    and multi-profile files (one profile per wind cluster). For
    multiple profiles, the cluster ``probability_weight`` is used to
    compute the weighted average power at each wind speed; wind speeds
    where a profile has ``successful: false`` contribute zero power.
    The cycle time is averaged over the successful profiles only.

    Note:
        Wind speed entries are matched to ``reference_wind_speeds`` by
        position rather than by value, so reordered or partial
        profiles are mis-parsed.

    Args:
        power_curves_data (dict): Parsed contents of an awesIO power
            curves YAML file (``power_curves_schema.yml``).

    Raises:
        ValueError: If the file contains no power curves or a profile
            length does not match ``reference_wind_speeds``.

    Returns:
        dict: Dictionary with keys ``'windRange'`` [m/s], ``'peAvg'``
        [W], ``'dtCycle'`` [s], and ``'peRated'`` [W].
    """
    windRange = np.asarray(power_curves_data['reference_wind_speeds'],
                           dtype=float)
    curves = power_curves_data['power_curves']
    if not curves:
        raise ValueError("Power curves file contains no power curves.")

    nSpeeds = len(windRange)
    powerSum = np.zeros(nSpeeds)
    timeSum = np.zeros(nSpeeds)
    weightTotal = 0.0
    timeWeight = np.zeros(nSpeeds)

    isSingleProfile = len(curves) == 1
    for curve in curves:
        weight = 1.0 if isSingleProfile else float(
            curve.get('probability_weight', 1.0))
        entries = curve['wind_speed_data']
        if len(entries) != nSpeeds:
            raise ValueError(
                f"Profile {curve.get('profile_id')} has {len(entries)} "
                f"wind speed entries, expected {nSpeeds}."
            )
        weightTotal += weight
        for i, entry in enumerate(entries):
            if not entry.get('successful', False):
                continue
            performance = entry['performance']
            powerSum[i] += weight * performance['power']['average_cycle_power']
            timeSum[i] += weight * performance['timing']['cycle_time']
            timeWeight[i] += weight

    peAvg = powerSum / weightTotal
    with np.errstate(divide='ignore', invalid='ignore'):
        dtCycle = np.where(timeWeight > 0, timeSum / timeWeight, 0.0)

    return {
        'windRange': windRange,
        'peAvg': peAvg,
        'dtCycle': dtCycle,
        'peRated': float(np.max(peAvg)),
    }


def _to_builtin(value):
    """Recursively convert numpy types to plain Python types.

    Used to make the results structure YAML-serializable.

    Args:
        value: Arbitrary nested structure of dicts, lists, numpy
            arrays and scalars.

    Returns:
        The same structure using only built-in Python types.
    """
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class EcoMoEconomicModel(EconomicModel):
    """ECOMo economic model for airborne wind energy systems.

    Computes subsystem costs (kite, tether, ground station, BoS, BoP)
    and economic metrics (LCoE, NPV, IRR, ...) for GG/FG and
    fixed/soft-wing AWE system topologies. All inputs are read from
    YAML files; system performance data come either from a performance
    YAML (standalone mode) or from AWESPA output files
    (AWESPA-connected mode).
    """

    def __init__(self):
        """Initialize the ECOMo economic model."""
        self.settings: Optional[Dict[str, Any]] = None
        self.systemData: Optional[Dict[str, Any]] = None
        self.costInputsPath: Optional[Path] = None
        self.performancePath: Optional[Path] = None
        self.awespaData: Optional[Dict[str, Any]] = None
        self.windResource: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.inputs: Optional[EcoInputs] = None
        self.costs: Optional[EcoCosts] = None
        self.eco: Optional[Dict[str, Any]] = None

    def load_configuration(
        self,
        economic_settings_path: Path,
        validate: bool = True,
        **kwargs,
    ) -> None:
        """Load economic model configuration from YAML files.

        Reads the economic settings file and the awesIO system file it
        references. The cost inputs file and the performance source
        (the performance YAML, or the AWESPA output files when
        ``input_files.aep_results`` and ``input_files.power_curves``
        are both set) are loaded when :meth:`compute_economics` runs.
        All file references are resolved relative to the settings
        file's directory.

        Args:
            economic_settings_path (Path): Path to the economic
                settings YAML file (awesIO format with topology,
                input_files, wind_resource, business, replacements and
                system_extras sections).
            validate (bool): If True, attempt awesIO validation of the
                input files. No economic settings schema is available
                yet, so validation failures of the settings file are
                reported but not fatal. Defaults to True.

        Raises:
            FileNotFoundError: If the settings or system file does not
                exist.
            ValueError: If required sections are missing, the topology
                specifies an unknown power or wing type, or only one of
                the AWESPA output files is set.
        """
        # data_dir is a removed, deprecated parameter; input files are
        # resolved relative to the settings file.
        data_dir = kwargs.pop('data_dir', None)
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments: {', '.join(kwargs)}")
        if data_dir is not None:
            warnings.warn(
                "data_dir is deprecated and ignored; input files are "
                "resolved relative to the settings file.",
                DeprecationWarning,
            )

        settingsPath = Path(economic_settings_path)
        if not settingsPath.exists():
            raise FileNotFoundError(
                f"Economic settings file not found: {settingsPath}"
            )

        with open(settingsPath, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)

        # No economic settings schema is available yet, so validation
        # failures are not fatal. TODO: make validation strict once
        # awesIO provides an economic_settings_schema.yml.
        if validate and awesio_validate is not None:
            try:
                awesio_validate(settings)
            except Exception as e:
                print(f"Note: awesIO validation skipped for "
                      f"{settingsPath.name}: {e}")

        topology = settings.get('topology')
        if topology is None:
            raise ValueError(
                f"Missing 'topology' section in {settingsPath}"
            )
        if topology.get('power') not in VALID_POWER_TYPES:
            raise ValueError(
                f"topology.power must be one of {VALID_POWER_TYPES}, "
                f"got: {topology.get('power')}"
            )
        if topology.get('wing') not in VALID_WING_TYPES:
            raise ValueError(
                f"topology.wing must be one of {VALID_WING_TYPES}, "
                f"got: {topology.get('wing')}"
            )

        inputFiles = settings.get('input_files')
        if not inputFiles:
            raise ValueError(
                f"Missing 'input_files' section in {settingsPath}"
            )
        for key in ('system', 'cost_inputs'):
            if not inputFiles.get(key):
                raise ValueError(
                    f"Missing 'input_files.{key}' in {settingsPath}"
                )

        settingsDir = settingsPath.parent
        systemPath = settingsDir / inputFiles['system']
        self.costInputsPath = settingsDir / inputFiles['cost_inputs']

        # AWESPA-connected mode requires both output files to be set
        aepResults = inputFiles.get('aep_results')
        powerCurves = inputFiles.get('power_curves')
        if (aepResults is None) != (powerCurves is None):
            raise ValueError(
                "input_files.aep_results and input_files.power_curves "
                "must both be set or both be null"
            )

        self.settings = settings
        self.awespaData = None
        self.performancePath = None
        if aepResults is not None:
            self._load_awespa_data(settingsDir / aepResults,
                                   settingsDir / powerCurves, validate)
        else:
            self.performancePath = (settingsDir / inputFiles['performance']
                                    if inputFiles.get('performance') else None)

        self.systemData = eco_load_system(systemPath,
                                          wing=topology['wing'],
                                          validate=validate)

        # Wind speed distribution: the site wind resource file when
        # provided, otherwise the Weibull parameters in the settings
        windResourceCfg = settings.get('wind_resource') or {}
        resourceFile = windResourceCfg.get('resource_file')
        self.windResource = None
        if resourceFile is not None:
            self.windResource = eco_load_wind_resource(
                settingsDir / resourceFile, validate=validate)

        print(f"Loaded ECOMo configuration: "
              f"{settings.get('metadata', {}).get('name', settingsPath.name)}")
        print(f"  Topology:    {topology['power']} / {topology['wing']} wing")
        print(f"  System:      {inputFiles['system']}")
        print(f"  Cost inputs: {inputFiles['cost_inputs']}")
        if self.awespaData is not None:
            print(f"  AEP results: {aepResults}")
            print(f"  Power curves:{powerCurves}")
        else:
            print(f"  Performance: {inputFiles.get('performance')}")
        if self.windResource is not None:
            print(f"  Wind resource: {resourceFile}")
        else:
            print(f"  Wind resource: Weibull "
                  f"(k={windResourceCfg.get('weibull_shape', WEIBULL_SHAPE)}, "
                  f"A={windResourceCfg.get('weibull_scale', WEIBULL_SCALE)})")

    def _load_awespa_data(
        self,
        aep_path: Path,
        power_curves_path: Path,
        validate: bool,
    ) -> None:
        """Load AEP and power curve data from AWESPA output files.

        Switches the model to AWESPA-connected mode: the wind range,
        average and rated electrical power, cycle time and AEP are
        taken from the AWESPA outputs and the performance file is
        ignored.

        Args:
            aep_path (Path): Path to an AWESPA ``aep_results.yml``
                file (output of ``calculate_aep()``).
            power_curves_path (Path): Path to an awesIO power curves
                YAML file (``power_curves_schema.yml``).
            validate (bool): If True, attempt awesIO validation of the
                power curves file.

        Raises:
            FileNotFoundError: If either file does not exist.
            ValueError: If required fields are missing.
        """
        aepPath = Path(aep_path)
        powerCurvesPath = Path(power_curves_path)
        for label, path in [("AEP results", aepPath),
                            ("Power curves", powerCurvesPath)]:
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")

        with open(aepPath, 'r', encoding='utf-8') as f:
            aepData = yaml.safe_load(f)
        with open(powerCurvesPath, 'r', encoding='utf-8') as f:
            powerCurvesData = yaml.safe_load(f)

        if validate and awesio_validate is not None:
            try:
                awesio_validate(powerCurvesData)
            except Exception as e:
                print(f"Note: awesIO validation skipped for "
                      f"{powerCurvesPath.name}: {e}")

        if 'total_aep' not in aepData or 'kwh' not in aepData['total_aep']:
            raise ValueError(
                f"Missing 'total_aep.kwh' field in {aepPath}"
            )

        parsed = _parse_power_curves(powerCurvesData)
        # AEP is stored in MWh, matching the units used in eco_metrics
        parsed['aepMwh'] = float(aepData['total_aep']['kwh']) / 1e3

        # Prefer the rated power reported with the AEP results (max
        # power across all profiles) over the weighted-average maximum.
        ratedPowerKw = aepData.get('rated_power_kw')
        if ratedPowerKw is not None:
            parsed['peRated'] = float(ratedPowerKw) * 1e3

        self.awespaData = parsed

    def compute_economics(
        self,
        output_path: Path = None,
        verbose: bool = False,
        showplot: bool = False,
        saveplot: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Run the ECOMo model and optionally export results to YAML.

        Builds the typed input and cost structures from the configured
        YAML files (or the AWESPA outputs when loaded), runs all
        subsystem cost modules and computes the economic metrics.

        Args:
            output_path (Path): Path where the results YAML will be
                written. If None, no export is performed. Defaults to
                None.
            verbose (bool): Whether to print a summary of the computed
                metrics. Defaults to False.
            showplot (bool): Whether to display the cost-breakdown and
                cashflow figures. Defaults to False.
            saveplot (bool): Whether to save the figures next to the
                output file. Requires ``output_path``. Defaults to
                False.
            validate (bool): If True, attempt awesIO validation of the
                output file. No economic results schema is available
                yet, so validation failures are reported but not
                fatal. Defaults to True.

        Raises:
            ValueError: If the model configuration has not been loaded
                or no performance data source is available.

        Returns:
            dict: The ``eco`` results structure including the
            ``metrics`` section (LCoE, NPV, ICC, ...).
        """
        if self.settings is None:
            raise ValueError(
                "Model not configured. Call load_configuration first."
            )

        topology = Topology(power=self.settings['topology']['power'],
                            wing=self.settings['topology']['wing'])

        # Performance data (and the per-storage exchanged energy)
        if self.awespaData is not None:
            performance, exchangedEnergy, forceAvailable = (
                self._awespa_performance(topology))
        else:
            if self.performancePath is None:
                raise ValueError(
                    "No performance data: set 'input_files.performance', "
                    "or both 'input_files.aep_results' and "
                    "'input_files.power_curves', in the settings."
                )
            performance, exchangedEnergy, forceAvailable = (
                self._standalone_performance(topology))

        # Typed inputs and costs
        inputs = EcoInputs(
            topology=topology,
            business=self._build_business(),
            kite=self._build_kite(forceAvailable),
            tether=self._build_tether(),
            groundStation=self._build_ground_station(topology, exchangedEnergy),
            performance=performance,
        )
        costs = eco_load_cost_inputs(
            self.costInputsPath,
            tether_max_stress=self.systemData['tetherMaxStress'],
            power=topology.power)

        # Run the simulation
        eco = eco_main(inputs, costs)
        self.inputs, self.costs, self.eco = inputs, costs, eco

        if output_path is not None:
            self._export_results(Path(output_path), eco, validate)

        if verbose:
            self._print_summary(eco)

        if showplot or saveplot:
            saveDir = (str(Path(output_path).parent)
                       if (saveplot and output_path is not None) else None)
            eco_display_results(eco, show=showplot, save_dir=saveDir)

        return eco

    # ------------------------------------------------------------------ #
    #  Input assembly helpers
    # ------------------------------------------------------------------ #

    def _replacement(self, name: str) -> Optional[float]:
        """Read a replacement frequency from the settings.

        Args:
            name (str): Key in the settings ``replacements`` section.

        Returns:
            float: Replacement frequency [1/year], or None (= auto
            estimate) when the key is null or absent.
        """
        value = (self.settings.get('replacements') or {}).get(name)
        return None if value is None else float(value)

    def _wind_distribution(self, windRange: np.ndarray) -> np.ndarray:
        """Evaluate the wind speed probability density.

        When a wind resource file is configured, the site density is
        interpolated onto ``windRange``; otherwise the Weibull
        parameters from the settings are used.

        Args:
            windRange (np.ndarray): Wind speeds [m/s].

        Returns:
            np.ndarray: Probability density at each wind speed.
        """
        if self.windResource is not None:
            binCenters, density = self.windResource
            return np.interp(windRange, binCenters, density)
        windResource = self.settings.get('wind_resource') or {}
        return weibull_pdf(
            windRange,
            shape=windResource.get('weibull_shape', WEIBULL_SHAPE),
            scale=windResource.get('weibull_scale', WEIBULL_SCALE),
        )

    def _resolve_tether_force(self, value,
                              windRange: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Resolve the tether force to a per-wind-speed array.

        The absence of a tether force is warned about at its source (the
        performance loader in standalone mode, or the override check in
        AWESPA-connected mode), so this conversion is silent.

        Args:
            value: Per-wind-speed array, scalar, or None.
            windRange (np.ndarray): Wind speeds [m/s].

        Raises:
            ValueError: If an array does not match the wind range.

        Returns:
            tuple: ``(tetherForce, available)`` where ``available`` is
            False when no tether force was provided (the force is then
            an array of zeros).
        """
        if value is None:
            return np.zeros_like(windRange), False
        if np.isscalar(value):
            return float(value) * np.ones_like(windRange), True
        tetherForce = np.asarray(value, dtype=float)
        if len(tetherForce) != len(windRange):
            raise ValueError(
                f"The tether force has {len(tetherForce)} entries but "
                f"the wind range has {len(windRange)} points."
            )
        return tetherForce, True

    def _standalone_performance(
        self, topology: Topology,
    ) -> Tuple[PerformanceData, Dict[str, Any], bool]:
        """Build the performance data from the performance YAML file.

        Args:
            topology (Topology): System topology.

        Raises:
            ValueError: If a field required for the topology is
                missing.

        Returns:
            tuple: ``(performance, exchangedEnergy, forceAvailable)``.
        """
        perf = eco_load_performance(self.performancePath, topology.power)
        windRange = perf['windRange']
        tetherForce, available = self._resolve_tether_force(
            perf['tetherForce'], windRange)

        peakMechanicalPower = perf['pmPeak']
        cycleTime = perf['dtCycle']
        if topology.power == 'GG' and peakMechanicalPower is None:
            peakMechanicalPower = peak_mechanical_power_fallback(
                perf['peRated'])

        turningRadius = perf['turningRadius']
        if (topology.power == 'FG' and turningRadius is None and
                self.systemData['kiteSpan'] is not None):
            turningRadius = TURNING_RADIUS_TO_SPAN * self.systemData['kiteSpan']

        performance = PerformanceData(
            windSpeeds=windRange,
            windPdf=self._wind_distribution(windRange),
            averagePower=perf['peAvg'],
            ratedPower=perf['peRated'],
            tetherForce=tetherForce,
            peakMechanicalPower=peakMechanicalPower,
            cycleTime=cycleTime,
            tipSpeedRatio=perf['tipSpeedRatio'],
            turningRadius=turningRadius,
            externalAep=None,
        )
        return performance, perf['exchangedEnergy'], available

    def _awespa_performance(
        self, topology: Topology,
    ) -> Tuple[PerformanceData, Dict[str, Any], bool]:
        """Build the performance data from the loaded AWESPA outputs.

        The performance file from the settings is ignored; the tether
        force comes from ``system_extras.tether_force_override`` and
        the storage exchanged energy falls back to half the rated
        capacity.

        Args:
            topology (Topology): System topology.

        Returns:
            tuple: ``(performance, exchangedEnergy, forceAvailable)``.
        """
        data = self.awespaData
        windRange = data['windRange']
        override = (self.settings.get('system_extras') or {}).get(
            'tether_force_override')
        if override is None:
            warnings.warn(
                "No tether force available; tether force is not provided "
                "by the AWESPA power model outputs. Tether and soft-kite "
                "replacement costs will not be estimated. Set "
                "'system_extras.tether_force_override' in the settings "
                "YAML to enable these calculations.",
                UserWarning, stacklevel=2,
            )
        tetherForce, available = self._resolve_tether_force(override, windRange)

        peakMechanicalPower = (peak_mechanical_power_fallback(data['peRated'])
                               if topology.power == 'GG' else None)

        performance = PerformanceData(
            windSpeeds=windRange,
            windPdf=self._wind_distribution(windRange),
            averagePower=data['peAvg'],
            ratedPower=data['peRated'],
            tetherForce=tetherForce,
            peakMechanicalPower=peakMechanicalPower,
            cycleTime=data['dtCycle'],
            tipSpeedRatio=None,
            turningRadius=None,
            externalAep=data['aepMwh'],
        )
        return performance, {}, available

    def _build_business(self) -> BusinessInputs:
        """Build the business inputs from the settings."""
        business = self.settings['business']
        return BusinessInputs(
            nYears=int(business['n_years']),
            costOfDebt=float(business['cost_of_debt']),
            costOfEquity=float(business['cost_of_equity']),
            taxRate=float(business['tax_rate']),
            debtToEquity=float(business['debt_to_equity']),
        )

    def _build_kite(self, force_available: bool) -> KiteInputs:
        """Build the kite inputs from the system YAML and settings.

        Args:
            force_available (bool): Whether a tether force is available;
                if not, an auto-estimated structure replacement
                frequency is disabled (it would divide by zero force).

        Returns:
            KiteInputs: The kite parameters.
        """
        extras = self.settings.get('system_extras') or {}
        replacement = self._replacement('kite_structure')
        if not force_available and replacement is None:
            replacement = 0.0
        return KiteInputs(
            mass=self.systemData['kiteMass'],
            flatArea=self.systemData['kiteFlatArea'],
            structureReplacementFrequency=replacement,
            onboardGeneratorPower=extras.get('onboard_generator_rated_power'),
            onboardBatteryCapacity=extras.get('onboard_battery_capacity'),
        )

    def _build_tether(self) -> TetherInputs:
        """Build the tether inputs from the system YAML and settings."""
        tether = self.systemData['tether']
        return TetherInputs(
            diameter=tether['d'],
            length=tether['L'],
            density=tether['rho'],
            replacementFrequency=self._replacement('tether'),
        )

    def _build_storage(self, name: str, capacity: float, power: str,
                       exchangedEnergy: Dict[str, Any]) -> StorageInputs:
        """Build one storage bank input, applying the E_ex fallback.

        Args:
            name (str): Storage name in the settings/performance files.
            capacity (float): Rated storage capacity [kWh].
            power (str): Power generation type, 'GG' or 'FG'.
            exchangedEnergy (dict): Exchanged energy per storage name
                [kWh].

        Returns:
            StorageInputs: The storage bank parameters.
        """
        exchanged = exchangedEnergy.get(name)
        if exchanged is None and power == 'GG':
            warnings.warn(
                f"No exchanged energy data for '{name}'; using half the "
                f"rated capacity.",
                UserWarning,
            )
            exchanged = capacity / 2
        return StorageInputs(
            ratedCapacity=capacity,
            exchangedEnergy=exchanged,
            replacementFrequency=self._replacement(name),
        )

    def _build_ground_station(self, topology: Topology,
                              exchangedEnergy: Dict[str, Any]
                              ) -> GroundStationInputs:
        """Build the ground station inputs.

        Args:
            topology (Topology): System topology.
            exchangedEnergy (dict): Exchanged energy per storage name
                [kWh].

        Returns:
            GroundStationInputs: The ground station parameters.
        """
        extras = self.settings.get('system_extras') or {}
        replacements = self.settings.get('replacements') or {}
        capacities = self.systemData['storageCapacities']

        ultracapacitor = None
        if 'ultracapacitor' in capacities:
            ultracapacitor = self._build_storage(
                'ultracapacitor', capacities['ultracapacitor'],
                topology.power, exchangedEnergy)
        battery = None
        if 'battery' in capacities:
            battery = self._build_storage(
                'battery', capacities['battery'],
                topology.power, exchangedEnergy)

        hydraulicAccumulator = None
        accumulatorCapacity = extras.get('hydraulic_accumulator_capacity')
        if accumulatorCapacity is not None:
            hydraulicAccumulator = self._build_storage(
                'hydraulic_accumulator', float(accumulatorCapacity),
                topology.power, exchangedEnergy)

        return GroundStationInputs(
            ultracapacitor=ultracapacitor,
            battery=battery,
            hydraulicAccumulator=hydraulicAccumulator,
            hydraulicMotorReplacementFrequency=(
                self._replacement('hydraulic_motor')
                if 'hydraulic_motor' in replacements else None),
            pumpMotorReplacementFrequency=(
                self._replacement('pump_motor')
                if 'pump_motor' in replacements else None),
        )

    # ------------------------------------------------------------------ #
    #  Output helpers
    # ------------------------------------------------------------------ #

    def _export_results(self, outputPath: Path, eco: Dict[str, Any],
                        validate: bool) -> None:
        """Write the economic results to an awesIO-style YAML file.

        Args:
            outputPath (Path): Destination file path.
            eco (dict): Economic results structure.
            validate (bool): If True, attempt awesIO validation of the
                written file (non-fatal).
        """
        metrics = eco['metrics']
        settingsMeta = self.settings.get('metadata', {})

        results = {
            'metadata': {
                'name': f"{settingsMeta.get('name', 'ECOMo')} - Results",
                'description': "Economic metrics and cost breakdown "
                               "computed by the ECOMo economic model",
                'note': ("System performance data taken from AWESPA "
                         "output files"
                         if self.awespaData is not None else
                         "System performance data taken from "
                         "standalone inputs"),
                'awesIO_version': AWESIO_VERSION,
                # schema omitted: no economic_results_schema.yml in awesIO yet
                'time_created': datetime.datetime.now().isoformat(),
            },
            'topology': {
                'power': self.settings['topology']['power'],
                'wing': self.settings['topology']['wing'],
            },
            'metrics': {
                'lcoe_eur_per_mwh': metrics['LCoE'],
                'cove_eur_per_mwh': metrics['CoVE'],
                'lroe_eur_per_mwh': metrics['LRoE'],
                'lpoe_eur_per_mwh': metrics['LPoE'],
                'icc_eur': metrics['ICC'],
                'omc_eur_per_year': metrics['OMC'],
                'npv_eur': metrics['NPV'],
                'irr': metrics['IRR'],
                'aep_mwh': metrics['AEP'],
                'capacity_factor': metrics['CF'],
                'capital_recovery_factor': metrics['CRF'],
                'electricity_price_eur_per_mwh': metrics['p'],
                'value_factor': metrics['vf'],
                'payback_year': metrics['payback_year'],
                'annual_profit_eur': metrics['Pi'],
            },
            'cost_breakdown': {
                'capex_eur': dict(zip(metrics['icc_name'],
                                      metrics['icc'])),
                'opex_eur_per_year': dict(zip(metrics['omc_name'],
                                              metrics['omc'])),
                'lcoe_contributions_eur_per_mwh': dict(zip(
                    metrics['LCoE_contr_name'], metrics['LCoE_contr'])),
            },
            'cashflow_eur': metrics['cashflow'],
        }

        outputPath.parent.mkdir(parents=True, exist_ok=True)
        with open(outputPath, 'w', encoding='utf-8') as f:
            yaml.safe_dump(_to_builtin(results), f, sort_keys=False,
                           default_flow_style=False)

        # No economic results schema is available yet, so validation
        # failures are not fatal. TODO: make validation strict once
        # awesIO provides an economic_results_schema.yml.
        if validate and awesio_validate is not None:
            try:
                awesio_validate(str(outputPath))
            except Exception as e:
                print(f"Note: awesIO validation skipped for "
                      f"{outputPath.name}: {e}")

        print(f"Results written to {outputPath}")

    @staticmethod
    def _print_summary(eco: Dict[str, Any]) -> None:
        """Print a compact summary of the computed metrics.

        Args:
            eco (dict): Economic results structure.
        """
        metrics = eco['metrics']
        print("\nECOMo economic metrics:")
        print(f"  ICC  = {metrics['ICC'] / 1e3:.1f} k EUR")
        print(f"  OMC  = {metrics['OMC'] / 1e3:.1f} k EUR/year")
        print(f"  AEP  = {metrics['AEP']:.2f} MWh")
        print(f"  CF   = {metrics['CF']:.3f}")
        print(f"  LCoE = {metrics['LCoE']:.1f} EUR/MWh")
        print(f"  CoVE = {metrics['CoVE']:.1f} EUR/MWh")
        print(f"  LRoE = {metrics['LRoE']:.1f} EUR/MWh")
        print(f"  LPoE = {metrics['LPoE']:.1f} EUR/MWh")
        print(f"  NPV  = {metrics['NPV'] / 1e3:.1f} k EUR")
        if np.isnan(metrics['IRR']):
            print("  IRR  = undefined (cash flows never break even)")
        else:
            print(f"  IRR  = {metrics['IRR']:.4f}")
        print(f"  Payback year = {metrics['payback_year']}")

"""Unit tests for the subsystem edge cases (tether, gStation, metrics)."""

import numpy as np
import pytest

from ecomo.subsystems.eco_tether import (
    eco_tether,
    _cycles_to_failure,
    _bending_replacement_frequency,
)
from ecomo.subsystems.eco_kite import _reelout_hour_replacement_frequency
from ecomo.subsystems.eco_gstation import (
    _gg_storage_replacement_frequency,
    _life_replacement_frequency,
)
from ecomo.subsystems.eco_bos import (
    eco_bos,
    _operation_maintenance_labour,
)
from ecomo.eco_hours import (
    annual_cycle_count,
    annual_flight_hours,
    annual_reelout_hours,
)
from ecomo.eco_metrics import eco_irr, eco_npv
from ecomo.eco_costs import (
    BalanceOfSystemCosts,
    KiteCosts,
    StorageCosts,
    TetherCosts,
)
from ecomo.eco_inputs import (
    BusinessInputs,
    OperationsInputs,
    PerformanceData,
    StorageInputs,
    TetherInputs,
    Topology,
)


def _performance(**overrides):
    """Build a minimal PerformanceData, with optional field overrides."""
    fields = dict(
        windSpeeds=np.array([5.0, 10.0]),
        windPdf=np.array([0.10, 0.05]),
        averagePower=np.array([1e5, 5e5]),
        ratedPower=1e6,
        tetherForce=np.array([1e4, 2e4]),
        cycleTime=np.array([100.0, 100.0]),
    )
    fields.update(overrides)
    return PerformanceData(**fields)


def _tether_costs(operational_life=None, **master_curve):
    return TetherCosts(
        priceMass=80, fibreAreaFraction=0.85, coatingMassFraction=0.1,
        maxStress=1.5e9,
        creepLifeCoefficients=np.array([-2.4, 8.3, -11.2, 5.2]),
        conductiveManufacturingFactor=2.5,
        bendingLifeA1=6.5, bendingLifeA2=2.6, nBends=2,
        operationalLife=operational_life,
        **master_curve,
    )


_MASTER_CURVE = dict(masterCurveCoeff=5.0e6, masterCurveExponent=2.0,
                     bearingPressureCoeff=1.0, pwLimitMpa=8.0,
                     bendingDdRatio=30.0, designSafetyFactor=3.0)


def test_master_curve_cycles_to_failure():
    # p_N = k_pw * sigma_MPa / (D/d); N_f = C * p_N^(-B).
    # sigma = 600 MPa, D/d = 30 -> p_N = 20 MPa -> N_f = 5e6 * 20^-2 = 12500
    costs = _tether_costs(**_MASTER_CURVE)
    nFail = _cycles_to_failure(np.array([600e6]), costs)
    assert nFail[0] == pytest.approx(5.0e6 * 20.0 ** -2.0)


def test_master_curve_low_pressure_clamp():
    # sigma = 30 MPa -> p_N = 1 MPa, below the 8 MPa clamp -> evaluated at 8
    costs = _tether_costs(**_MASTER_CURVE)
    nFail = _cycles_to_failure(np.array([30e6]), costs)
    assert nFail[0] == pytest.approx(5.0e6 * 8.0 ** -2.0)


def test_master_curve_preferred_over_a1a2_when_set():
    # With the master curve configured, N_f follows it, not the a1/a2 model
    withMc = _cycles_to_failure(np.array([600e6]), _tether_costs(**_MASTER_CURVE))
    withoutMc = _cycles_to_failure(np.array([600e6]), _tether_costs())
    assert withMc[0] == pytest.approx(5.0e6 * 20.0 ** -2.0)
    assert withoutMc[0] != pytest.approx(withMc[0])  # a1/a2 semi-log differs


def test_master_curve_safety_factor_scales_frequency():
    perf = _performance()
    stress = np.array([4e8, 4e8])
    withSf = _bending_replacement_frequency(
        stress, perf, _tether_costs(**_MASTER_CURVE))
    noSf = _bending_replacement_frequency(
        stress, perf, _tether_costs(**{**_MASTER_CURVE,
                                       'designSafetyFactor': None}))
    assert withSf == pytest.approx(3.0 * noSf)


def test_safety_factor_ignored_without_master_curve():
    # SF is a master-curve derating; it must NOT scale the a1/a2 fallback
    perf = _performance()
    stress = np.array([4e8, 4e8])
    a1a2 = _bending_replacement_frequency(stress, perf, _tether_costs())
    a1a2WithSfField = _bending_replacement_frequency(
        stress, perf, _tether_costs(designSafetyFactor=3.0))
    assert a1a2WithSfField == pytest.approx(a1a2)


def _business():
    return BusinessInputs(nYears=25, costOfDebt=0.08, costOfEquity=0.12,
                          taxRate=0.25, debtToEquity=70 / 30)


def _bos_costs(om_price_power=60.0):
    return BalanceOfSystemCosts(
        sitePreparationPricePower=40.0,
        foundationPricePower=55.0,
        installationPricePower=40.0,
        operationsMaintenancePricePower=om_price_power,
        decommissioningInstallationFraction=0.5,
    )


def _operating_days(perf):
    """Expected N_op = f_wind * 365 for a performance case."""
    from ecomo.eco_hours import annual_operating_days
    return annual_operating_days(perf)


def _flight_hours(perf, availability=1.0):
    """Expected annual flight hours for a performance case."""
    from ecomo.eco_hours import annual_flight_hours
    return annual_flight_hours(perf, availability)


def test_operation_labour_is_operating_day_based():
    perf = _performance()
    ops = OperationsInputs(labourPrice=50.0, operatingHoursPerDay=2.0,
                           maintenanceHoursPerFlightHour=0.5, availability=1.0)
    labour = _operation_maintenance_labour(ops, perf)

    # C_op = (1 - automation) * w_lab * N_op * h_op (base crew, per day)
    nOp = _operating_days(perf)
    assert labour['operating_days'] == pytest.approx(nOp)
    assert labour['operation'] == pytest.approx(50.0 * nOp * 2.0)
    # C_maint = w_lab * h_maint_per_flight_h * H_flight (per flight hour)
    hFlight = _flight_hours(perf)
    assert labour['maintenance'] == pytest.approx(50.0 * 0.5 * hFlight)


def test_operation_labour_scales_with_automation():
    perf = _performance()
    base = dict(labourPrice=50.0, operatingHoursPerDay=2.0,
                maintenanceHoursPerFlightHour=0.5)
    manual = _operation_maintenance_labour(
        OperationsInputs(**base, automation=0.0), perf)
    half = _operation_maintenance_labour(
        OperationsInputs(**base, automation=0.5), perf)
    full = _operation_maintenance_labour(
        OperationsInputs(**base, automation=1.0), perf)

    # C_op = (1 - automation) * w_lab * N_op * h_op
    nOp = _operating_days(perf)
    assert manual['operation'] == pytest.approx(50.0 * nOp * 2.0)
    assert half['operation'] == pytest.approx(0.5 * manual['operation'])
    assert full['operation'] == 0.0
    # Maintenance labour is unaffected by the automation fraction
    assert half['maintenance'] == pytest.approx(manual['maintenance'])


def test_operation_labour_availability_dependence():
    # Base-crew operation labour is tied to the operating calendar, so it
    # is availability-independent; maintenance is now charged per flight
    # hour, so it scales down with availability (fewer hours flown).
    perf = _performance()
    base = dict(labourPrice=50.0, operatingHoursPerDay=2.0,
                maintenanceHoursPerFlightHour=0.5)
    full = _operation_maintenance_labour(
        OperationsInputs(**base, availability=1.0), perf, availability=1.0)
    down = _operation_maintenance_labour(
        OperationsInputs(**base, availability=0.5), perf, availability=0.5)
    assert down['operation'] == pytest.approx(full['operation'])
    assert down['maintenance'] == pytest.approx(0.5 * full['maintenance'])


def test_bos_om_overhead_and_separate_labour_leaf():
    ops = OperationsInputs(labourPrice=50.0, operatingHoursPerDay=2.0,
                           maintenanceHoursPerFlightHour=0.5, availability=1.0)
    perf = _performance()
    costs = _bos_costs(om_price_power=60.0)

    withLabour = eco_bos(perf, costs, ops)
    withoutLabour = eco_bos(perf, costs, None)

    overhead = 60.0 * perf.ratedPower / 1e3
    labour = _operation_maintenance_labour(ops, perf)
    # BoS.OM.OPEX is the per-kW overhead ONLY (no labour folded in)
    assert withLabour['OM']['OPEX'] == pytest.approx(overhead)
    assert withoutLabour['OM']['OPEX'] == pytest.approx(overhead)
    # The crew labour is its own separate leaf group
    assert withLabour['labour']['OPEX'] == pytest.approx(
        labour['operation'] + labour['maintenance'])
    # Without an operations block, there is no labour group at all
    assert 'labour' not in withoutLabour


def test_life_replacement_frequency_convention():
    # 20-yr component in a 25-yr project -> one twentieth per year
    assert _life_replacement_frequency(20.0, 25) == pytest.approx(1 / 20)
    # Life beyond the project, unset or zero -> no replacement charged
    assert _life_replacement_frequency(30.0, 25) == 0.0
    assert _life_replacement_frequency(None, 25) == 0.0
    assert _life_replacement_frequency(0.0, 25) == 0.0


def test_operations_maintenance_scales_with_flight_hours():
    ops = OperationsInputs(labourPrice=50.0, operatingHoursPerDay=2.0,
                           maintenanceHoursPerFlightHour=0.2, availability=1.0)
    perf = _performance()

    labour = _operation_maintenance_labour(ops, perf)
    hFlight = _flight_hours(perf)
    # C_maint = w_lab * h_maint_per_flight_h * H_flight
    assert labour['flight_hours'] == pytest.approx(hFlight)
    assert labour['maintenance'] == pytest.approx(0.2 * hFlight * 50.0)


def test_operating_days_exclude_zero_power_speeds():
    ops = OperationsInputs(labourPrice=50.0, operatingHoursPerDay=2.0,
                           maintenanceHoursPerFlightHour=0.5)
    # Below cut-in (zero power) wind speeds are not operating days
    perf = _performance(averagePower=np.array([0.0, 5e5]))
    fraction = np.trapezoid(perf.windPdf * np.array([0.0, 1.0]),
                            perf.windSpeeds)
    assert _operation_maintenance_labour(ops, perf)['operating_days'] == (
        pytest.approx(365.0 * fraction))


def test_eco_tether_accepts_zero_replacement_frequency():
    # A user-supplied replacement frequency of exactly 0 means no
    # replacement and must not raise ZeroDivisionError
    tether = TetherInputs(diameter=0.01, length=100.0, density=970.0,
                          replacementFrequency=0.0)
    eco = eco_tether(tether, _performance(), _tether_costs(), _business(),
                     Topology('GG', 'fixed'))

    assert eco['f_repl'] == 0
    assert eco['OPEX'] == 0
    assert eco['CAPEX'] > 0


def test_eco_tether_keeps_positive_replacement_frequency():
    tether = TetherInputs(diameter=0.01, length=100.0, density=970.0,
                          replacementFrequency=0.5)
    eco = eco_tether(tether, _performance(), _tether_costs(), _business(),
                     Topology('GG', 'fixed'))

    assert eco['f_repl'] == pytest.approx(0.5)
    assert eco['OPEX'] == pytest.approx(0.5 * eco['CAPEX'])


def test_eco_tether_auto_estimates_replacement_frequency():
    # None requests auto-estimation from the bending and creep models
    tether = TetherInputs(diameter=0.01, length=100.0, density=970.0,
                          replacementFrequency=None)
    eco = eco_tether(tether, _performance(), _tether_costs(), _business(),
                     Topology('GG', 'fixed'))

    assert 'f_repl_bend' in eco
    assert 'f_repl_creep' in eco
    assert np.isfinite(eco['f_repl'])


def test_storage_replacement_frequency_zero_cycle_time():
    # A scalar cycle time of zero means the system never cycles; the
    # replacement frequency must be 0, not an error
    storage = StorageInputs(ratedCapacity=10.0, exchangedEnergy=1.0)
    costs = StorageCosts(priceEnergy=60000, cycleLife=1e6)
    performance = _performance(cycleTime=0.0,
                               averagePower=np.array([0.0, 0.0]))

    assert _gg_storage_replacement_frequency(
        storage, costs, performance) == 0


def test_storage_replacement_frequency_scalar_inputs():
    storage = StorageInputs(ratedCapacity=10.0, exchangedEnergy=1.0)
    costs = StorageCosts(priceEnergy=60000, cycleLife=1e6)
    performance = _performance(cycleTime=100.0)

    expected = 8760 * np.trapezoid(
        performance.windPdf * (1.0 / 100.0 * 3600),
        performance.windSpeeds) / 10.0 / 1e6
    assert _gg_storage_replacement_frequency(
        storage, costs, performance) == pytest.approx(expected)


def _canopy_costs(canopy_life=250.0, per_cycle_penalty=None,
                  load_exponent=None, reference_force=None):
    return KiteCosts(avionicsCost=0.0,
                     canopyLifetimeFlightHours=canopy_life,
                     perCyclePenalty=per_cycle_penalty,
                     canopyLoadExponent=load_exponent,
                     canopyReferenceForce=reference_force)


def test_kite_reelout_hour_replacement_frequency():
    # f_repl = annual reel-out hours / canopy life (loaded hours)
    perf = _performance(reelOutTimeFraction=np.array([0.6, 0.6]))
    f = _reelout_hour_replacement_frequency(_canopy_costs(250.0), perf, 1.0)
    assert f == pytest.approx(annual_reelout_hours(perf, 1.0) / 250.0)


def test_kite_reelout_scales_with_availability():
    # Flying half as much halves the canopy replacement frequency
    perf = _performance(reelOutTimeFraction=np.array([0.6, 0.6]))
    full = _reelout_hour_replacement_frequency(_canopy_costs(), perf, 1.0)
    half = _reelout_hour_replacement_frequency(_canopy_costs(), perf, 0.5)
    assert half == pytest.approx(0.5 * full)


def test_kite_reelout_falls_back_to_flight_hours_without_timing():
    # No reel-out timing: count all flight hours as loaded, with a warning
    perf = _performance()  # reelOutTimeFraction is None
    with pytest.warns(UserWarning):
        f = _reelout_hour_replacement_frequency(_canopy_costs(250.0), perf, 1.0)
    assert f == pytest.approx(annual_flight_hours(perf, 1.0) / 250.0)


def test_kite_load_exponent_zero_matches_unweighted():
    # m = 0 (or None) reproduces the load-independent hour model exactly
    perf = _performance(reelOutTimeFraction=np.array([0.6, 0.6]),
                        tractionTetherForce=np.array([1e4, 2e4]))
    flat = _reelout_hour_replacement_frequency(_canopy_costs(), perf, 1.0)
    weighted0 = _reelout_hour_replacement_frequency(
        _canopy_costs(load_exponent=0.0), perf, 1.0)
    assert weighted0 == pytest.approx(flat)


def test_kite_load_weighting_matches_miner_rule():
    # f_repl = 8760 * integral(pdf * reel-out * (F/F_ref)^m) / canopy_life
    perf = _performance(reelOutTimeFraction=np.array([0.6, 0.6]),
                        tractionTetherForce=np.array([1e4, 2e4]))
    costs = _canopy_costs(250.0, load_exponent=2.0, reference_force=2e4)
    loadWeight = (np.array([1e4, 2e4]) / 2e4) ** 2.0        # [0.25, 1.0]
    expected = annual_reelout_hours(perf, 1.0, load_weight=loadWeight) / 250.0
    assert _reelout_hour_replacement_frequency(costs, perf, 1.0) == (
        pytest.approx(expected))


def test_kite_load_weighting_extends_life_below_reference():
    # Partial-load hours consume less life, so weighting lowers f_repl
    perf = _performance(reelOutTimeFraction=np.array([0.6, 0.6]),
                        tractionTetherForce=np.array([1e4, 2e4]))
    flat = _reelout_hour_replacement_frequency(_canopy_costs(), perf, 1.0)
    weighted = _reelout_hour_replacement_frequency(
        _canopy_costs(load_exponent=2.0, reference_force=2e4), perf, 1.0)
    assert weighted < flat


def test_kite_load_weighting_without_force_falls_back():
    # No tether force + a load exponent: warn and use unweighted hours
    perf = _performance(reelOutTimeFraction=np.array([0.6, 0.6]),
                        tetherForce=np.array([0.0, 0.0]))
    flat = _reelout_hour_replacement_frequency(_canopy_costs(), perf, 1.0)
    with pytest.warns(UserWarning, match="no tether force"):
        weighted = _reelout_hour_replacement_frequency(
            _canopy_costs(load_exponent=2.0), perf, 1.0)
    assert weighted == pytest.approx(flat)


def test_kite_per_cycle_penalty_adds_cycle_term():
    perf = _performance(reelOutTimeFraction=np.array([0.6, 0.6]))
    costs = _canopy_costs(250.0, per_cycle_penalty=1e-5)
    expected = (annual_reelout_hours(perf, 1.0) / 250.0 +
                1e-5 * annual_cycle_count(perf, 1.0))
    assert _reelout_hour_replacement_frequency(costs, perf, 1.0) == (
        pytest.approx(expected))


def test_tether_operational_life_governs_at_low_stress():
    # At the low stress of this case, bending/creep predict a near-infinite
    # life, so the empirical operational life governs the replacement
    tether = TetherInputs(diameter=0.01, length=100.0, density=970.0,
                          replacementFrequency=None)
    eco = eco_tether(tether, _performance(), _tether_costs(250.0),
                     _business(), Topology('GG', 'fixed'))

    expected = annual_flight_hours(_performance(), 1.0) / 250.0
    assert eco['f_repl_oper'] == pytest.approx(expected)
    assert eco['f_repl'] == pytest.approx(eco['f_repl_oper'])
    assert eco['f_repl'] > eco['f_repl_creep']
    assert eco['f_repl'] > eco['f_repl_bend']


def test_tether_operational_frequency_scales_with_availability():
    tether = TetherInputs(diameter=0.01, length=100.0, density=970.0,
                          replacementFrequency=None)
    full = eco_tether(tether, _performance(), _tether_costs(250.0),
                      _business(), Topology('GG', 'fixed'),
                      availability=1.0)['f_repl_oper']
    half = eco_tether(tether, _performance(), _tether_costs(250.0),
                      _business(), Topology('GG', 'fixed'),
                      availability=0.5)['f_repl_oper']
    assert half == pytest.approx(0.5 * full)


def test_tether_life_reports_governing_hours_operational_on():
    # Operational on (250 h) governs at low stress -> life = 250 flight h
    tether = TetherInputs(diameter=0.01, length=100.0, density=970.0,
                          replacementFrequency=None)
    eco = eco_tether(tether, _performance(), _tether_costs(250.0),
                     _business(), Topology('GG', 'fixed'))
    life = eco['life']
    assert life['governing_mode'] == 'operational'
    assert life['life_flight_hours'] == pytest.approx(250.0)
    assert life['operational_flight_hours'] == pytest.approx(250.0)
    # Life in years = life hours / annual flight hours
    assert life['life_years'] == pytest.approx(
        250.0 / life['annual_flight_hours'])


def test_tether_life_reports_bending_when_operational_off():
    # Operational off (None) -> bending governs at this low stress
    tether = TetherInputs(diameter=0.01, length=100.0, density=970.0,
                          replacementFrequency=None)
    eco = eco_tether(tether, _performance(), _tether_costs(None),
                     _business(), Topology('GG', 'fixed'))
    life = eco['life']
    assert life['governing_mode'] == 'bending'
    assert 'operational_flight_hours' not in life
    # Governing life matches the bending mode's life
    assert life['life_flight_hours'] == pytest.approx(
        life['bending_flight_hours'])


def _metrics(icc, omc, price, aep):
    return {'ICC': icc, 'OMC': omc, 'p': price, 'AEP': aep}


def test_eco_irr_root_of_npv():
    metrics = _metrics(icc=1e6, omc=1e4, price=100.0, aep=2000.0)
    irr = eco_irr(metrics, subsidy=0.0, nY=25)

    assert np.isfinite(irr)
    assert eco_npv(irr, metrics, 0.0, 25) == pytest.approx(0.0, abs=1e-3)


def test_eco_irr_undefined_when_revenue_negative():
    # Annual costs exceed revenues: the NPV never crosses zero
    metrics = _metrics(icc=1e6, omc=5e5, price=100.0, aep=2000.0)
    assert np.isnan(eco_irr(metrics, subsidy=0.0, nY=25))


def test_eco_irr_negative_rate_when_project_barely_recovers():
    # Total undiscounted revenue is below the investment: IRR < 0
    metrics = _metrics(icc=1e6, omc=0.0, price=100.0, aep=300.0)
    irr = eco_irr(metrics, subsidy=0.0, nY=25)

    assert np.isfinite(irr)
    assert irr < 0
    assert eco_npv(irr, metrics, 0.0, 25) == pytest.approx(0.0, abs=1e-3)

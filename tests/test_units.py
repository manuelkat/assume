from assume.common.base import SupportsMinMax, SupportsMinMaxCharge


def test_minmax():
    mm = SupportsMinMax("Test", "TestOperator", "TestTechnology", {}, None, "empty")
    mm.ramp_down = 200
    mm.ramp_up = 400
    mm.max_power = 1000
    mm.min_power = 200

    # stay turned off
    assert mm.calculate_ramp(previous_power=0, power=0, current_power=0) == 0
    # ramp up only possible to min_power
    assert mm.calculate_ramp(previous_power=0, power=190, current_power=0) == 200
    # ramp up 400
    assert mm.calculate_ramp(previous_power=0, power=1200, current_power=0) == 400
    # should not ramp up, if max_power already sold
    assert mm.calculate_ramp(previous_power=0, power=800, current_power=400) == 0
    # ramp up to 800
    assert mm.calculate_ramp(previous_power=400, power=1200, current_power=0) == 800
    # ramp up to max_power
    assert (
        mm.calculate_ramp(previous_power=800, power=1200, current_power=0)
        == mm.max_power
    )
    # can't sell more if already sold
    assert mm.calculate_ramp(previous_power=800, power=1200, current_power=1000) == 0

    # reduce output
    assert mm.calculate_ramp(previous_power=1000, power=800, current_power=0) == 800
    # use float
    assert mm.calculate_ramp(previous_power=800, power=753.2, current_power=0) == 753.2
    # check ramp down constraint
    assert mm.calculate_ramp(previous_power=800, power=500, current_power=0) == 600

    assert mm.calculate_ramp(previous_power=800, power=500, current_power=0) == 600


def test_minmaxcharge():
    mmc = SupportsMinMaxCharge(
        "Test", "TestOperator", "TestTechnology", {}, None, "empty"
    )

    mmc.ramp_down_charge = 100
    mmc.ramp_down_discharge = 100
    mmc.ramp_up_charge = 100
    mmc.ramp_up_discharge = 100
    mmc.max_power_charge = 1000
    mmc.max_power_discharge = 1000
    mmc.min_power_charge = 0
    mmc.min_power_discharge = 0

    # stay turned off
    assert (
        mmc.calculate_ramp_charge(previous_power=0, power_charge=0, current_power=0)
        == 0
    )

    # stay turned off
    assert (
        mmc.calculate_ramp_discharge(
            previous_power=0, power_discharge=0, current_power=0
        )
        == 0
    )
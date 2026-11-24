from solar.billing import energy_charge, monthly_bill
from solar.quote import monthly_savings, payback_years


def test_first_tier():
    assert energy_charge(80) == 8.0


def test_crosses_into_second_tier():
    assert round(energy_charge(250), 2) == 32.5


def test_tier_boundary_is_inclusive():
    assert round(energy_charge(100), 2) == 10.0
    assert round(energy_charge(300), 2) == 40.0


def test_top_tier():
    assert round(energy_charge(420), 2) == 64.0


def test_monthly_bill_includes_fixed_charge():
    assert monthly_bill(0) == 8.0
    assert monthly_bill(350) == 58.0


def test_savings_and_payback():
    usage = [400] * 12
    solar = [150] * 12
    assert monthly_savings(usage, solar)[0] == 27.5
    assert payback_years(9900, usage, solar) == 30.0

import pytest
from solar.billing import energy_charge, monthly_bill
from solar.quote import monthly_savings, payback_years


@pytest.mark.parametrize("kwh,cost", [(0, 0), (1, 0.1), (99.5, 9.95), (100, 10), (100.5, 10.075), (299, 39.85),
                                      (300, 40), (301, 40.2), (1000, 180)])
def test_energy(kwh, cost):
    assert energy_charge(kwh) == pytest.approx(cost)


def test_negative():
    with pytest.raises(ValueError):
        energy_charge(-1)


def test_bill_and_quote():
    assert monthly_bill(1000) == 188.0
    assert monthly_savings([90, 500], [200, 100]) == [9.0, 20.0]
    assert payback_years(100, [10] * 12, [0] * 12) == float("inf")

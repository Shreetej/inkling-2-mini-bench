from .tariffs import FIXED_MONTHLY_CHARGE, TIERS


def energy_charge(kwh: float) -> float:
    """Cost of `kwh` under the marginal tiers in tariffs.TIERS."""
    if kwh < 0:
        raise ValueError("kwh must be >= 0")
    for limit, rate in TIERS:
        if kwh < limit:
            return kwh * rate
    return 0.0


def monthly_bill(kwh: float) -> float:
    return round(FIXED_MONTHLY_CHARGE + energy_charge(kwh), 2)

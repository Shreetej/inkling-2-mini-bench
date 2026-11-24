from .billing import monthly_bill


def monthly_savings(usage_kwh: list[float], solar_kwh: list[float]) -> list[float]:
    """Bill without solar minus bill with solar, per month. Solar can't push usage below zero."""
    return [round(monthly_bill(u) - monthly_bill(max(u - s, 0)), 2) for u, s in zip(usage_kwh, solar_kwh)]


def payback_years(system_cost: float, usage_kwh: list[float], solar_kwh: list[float]) -> float:
    """Years to recover `system_cost` from one representative year of savings."""
    yearly = sum(monthly_savings(usage_kwh, solar_kwh))
    if yearly <= 0:
        return float("inf")
    return round(system_cost / yearly, 1)

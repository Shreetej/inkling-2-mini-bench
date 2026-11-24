"""Residential tariff: marginal tiers. Each tuple is (upper bound of the tier in kWh, price per kWh)."""
TIERS = [
    (100, 0.10),
    (300, 0.15),
    (float("inf"), 0.20),
]
FIXED_MONTHLY_CHARGE = 8.0

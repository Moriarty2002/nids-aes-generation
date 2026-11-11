"""Canonical padding schema for the active fixed-20 CICIoMT2024 packet adapter.

These are *invalid* real-packet coordinates. Occupancy is always a separate
boolean prefix mask, never inferred from these sentinel measurements.
"""
from types import MappingProxyType

CICIOMT2024_FIELDS = ("PL", "IAT", "DIR", "WIN", "FLG", "TTL")
CICIOMT2024_PAD_VALUES = MappingProxyType({
    "PL": 0.0, "IAT": 0.0, "DIR": -1.0,
    "WIN": -1.0, "FLG": -1.0, "TTL": 0.0,
})

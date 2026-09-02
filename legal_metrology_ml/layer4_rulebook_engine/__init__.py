from __future__ import annotations

"""
Layer 4: Rulebook Engine
Encapsulates all rule logic, thresholds, and condition evaluations.
"""

def __getattr__(name: str):
    if name == 'Engine':
        from .engine import Engine
        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['Engine']

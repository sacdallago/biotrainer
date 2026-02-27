from __future__ import annotations

from enum import Enum


class ViewMode(Enum):
    Leaderboard = "Leaderboard"
    Detailed = "Detailed"
    Compare = "Compare"


__all__ = ["ViewMode"]

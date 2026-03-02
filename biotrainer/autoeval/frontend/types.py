from __future__ import annotations

from enum import Enum


class ViewMode(Enum):
    Leaderboard = "Leaderboard"
    Detailed = "Detailed"
    Compare = "Compare"
    Evaluate = "Evaluate"
    Info = "Info"


__all__ = ["ViewMode"]

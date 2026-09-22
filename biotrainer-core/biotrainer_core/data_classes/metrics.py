from __future__ import annotations

from typing import Dict
from pydantic import BaseModel, Field

class EpochMetrics(BaseModel):
    epoch: int = Field(description="Epoch number")
    training: Dict = Field(description="Training metrics")
    validation: Dict = Field(description="Validation metrics")

    def to_dict(self):
        return {"epoch": self.epoch, "training": self.training, "validation": self.validation}


class MetricEstimate(BaseModel):
    name: str = Field(description="Name of the metric")
    mean: float = Field(description="Mean of the metric values")
    lower: float = Field(description="Lower bound of the metric values")
    upper: float = Field(description="Upper bound of the metric values")

    def overlaps_with(self, other: MetricEstimate) -> bool:
        """Check if the confidence intervals overlap with another estimate."""
        return self.lower <= other.upper and other.lower <= self.upper

    def __gt__(self, other: MetricEstimate) -> bool:
        """CI entirely above other's CI"""
        return self.lower > other.upper

    def __lt__(self, other: MetricEstimate) -> bool:
        """CI entirely below other's CI"""
        return self.upper < other.lower

    def __eq__(self, other: object) -> bool:
        """Return True if CIs overlap (no significant difference detected)."""
        if not isinstance(other, MetricEstimate):
            return NotImplemented
        if self.name != other.name:
            return False
        return self.overlaps_with(other)

    def get_error_margin(self) -> float:
        """Calculate the symmetric error margin from the confidence interval."""
        return (self.upper - self.lower) / 2.0

    def get_asymmetric_errors(self) -> tuple[float, float]:
        """Get asymmetric error bounds (lower_error, upper_error)."""
        return (self.mean - self.lower, self.upper - self.mean)

    def format_value(self, precision: int = 3) -> str:
        """
        Format the metric as 'Name: Mean +/- Error'.
        
        Args:
            precision: Number of decimal places to display
            
        Returns:
            Formatted string representation
        """
        error = self.get_error_margin()
        return f"{self.name}: {self.mean:.{precision}f} +/- {error:.{precision}f}"

    def format_value_with_ci(self, precision: int = 3) -> str:
        """
        Format the metric with explicit confidence interval bounds.
        
        Args:
            precision: Number of decimal places to display
            
        Returns:
            Formatted string with CI: 'Name: Mean +/- Error [Lower, Upper]'
        """
        error = self.get_error_margin()
        return (f"{self.name}: {self.mean:.{precision}f} +/- {error:.{precision}f} "
                f"[{self.lower:.{precision}f}, {self.upper:.{precision}f}]")

    def __str__(self) -> str:
        return self.format_value()

    def __repr__(self) -> str:
        """Detailed representation including confidence interval bounds."""
        return self.format_value_with_ci()


class BootstrappedMetric(MetricEstimate):
    iterations: int = Field(description="Number of iterations used for bootstrapping")
    sample_size: int = Field(description="Sample size used for bootstrapping")
    confidence_level: float = Field(description="Confidence level used for bootstrapping")

    def __eq__(self, other: object) -> bool:
        """Return True if CIs overlap (no significant difference detected)."""
        if not isinstance(other, MetricEstimate) or not isinstance(other, BootstrappedMetric):
            return NotImplemented
        if (self.iterations != other.iterations or
            self.sample_size != other.sample_size or
            self.confidence_level != other.confidence_level
        ):
            return False
        if self.name != other.name:
            return False
        return self.overlaps_with(other)
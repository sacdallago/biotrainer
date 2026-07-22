from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from .metrics import BootstrappedMetric


class ContactSingleProteinResult(BaseModel):
    """Required for bootstrapping and for caching and resuming from mid-dataset!"""
    protein_name: str = Field(description="Name of the protein")
    precision_scores: Dict[str, float] = Field(
        description="Precision scores for the protein's predicted contacts")  # e.g. {"short_P@L2": 0.83, "short_P@L5": 0.78, "long_P@L2": 0.81, "long_P@L5": 0.76}


class ContactDatasetResult(BaseModel):
    dataset_name: str = Field(description="Name of the dataset")
    aggregated_result: List[BootstrappedMetric] = Field(description="Aggregated and bootstrapped "
                                                                    "precision scores for the dataset")

    @classmethod
    def empty(cls, dataset_name: str):
        return ContactDatasetResult(dataset_name=dataset_name, aggregated_result=[])

    @classmethod
    def aggregate(cls,
                  dataset_name: str,
                  per_protein_results: List[ContactSingleProteinResult],
                  iterations: int = 30,  # Bootstrap iterations
                  seed: int = 42,  # Bootstrap seed
                  confidence_level: float = 0.05  # Bootstrap confidence level
                  ) -> ContactDatasetResult:
        from ..functions.bootstrapping import metrics_bootstrap
        if len(per_protein_results) == 0:
            return ContactDatasetResult.empty(dataset_name)

        first_value = per_protein_results[0]
        metric_names = list(first_value.precision_scores.keys())
        values = {metric_name: [p.precision_scores[metric_name] for p in per_protein_results]
                  for metric_name in metric_names}

        bt_res = metrics_bootstrap(
            metrics=values,
            iterations=iterations,
            sample_size=len(per_protein_results),
            seed=seed,
            confidence_level=confidence_level
        )

        return ContactDatasetResult(dataset_name=dataset_name, aggregated_result=bt_res)

    def __str__(self) -> str:
        scores = ", ".join(f"{metric.name}: {metric.mean:.3f}" for metric in self.aggregated_result)
        if not scores:
            return f"Dataset result [{self.dataset_name}] - No scores available"
        return f"Dataset result [{self.dataset_name}] - {scores}"

    def long_PatL2(self) -> Optional[float]:
        long_p_at_l2 = [metric for metric in self.aggregated_result if metric.name == "long_P@L2"]
        if len(long_p_at_l2) == 0:
            return None
        return long_p_at_l2[0].mean

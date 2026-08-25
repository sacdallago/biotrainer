from __future__ import annotations

import math
import numpy as np

from enum import Enum
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, field_validator, computed_field, SerializeAsAny

from .metrics import MetricEstimate, BootstrappedMetric


class ZeroShotMethod(Enum):
    WT_MARGINALS = "WT_MARGINALS"
    MASKED_MARGINALS = "MASKED_MARGINALS"
    PSEUDOPERPLEXITY = "PSEUDOPERPLEXITY"
    PERPLEXITY = "PERPLEXITY"
    JACOBIAN_CONTACT = "JACOBIAN_CONTACT"


class Mutation(BaseModel):
    """ Represents a single mutation (e.g. A15G)"""
    wt: str = Field(description="Wild type residue")
    position: int = Field(description="Position of the mutation")
    mt: str = Field(description="Mutated residue")

    one_indexed: Optional[bool] = Field(default=True, description="Whether the position is one-indexed or zero-indexed")

    @classmethod
    def parse(cls, mutation: str, one_indexed: Optional[bool] = True):
        if len(mutation) >= 3 and mutation[1:-1].isdigit():
            wt = mutation[0]
            mt = mutation[-1]
            pos = int(mutation[1:-1])
            if one_indexed and pos <= 0:
                raise ValueError(f"Position must be positive when one_indexed is True: {pos}")
            return cls(wt=wt, position=pos, mt=mt, one_indexed=one_indexed)

        # TODO Add multiple mutations support
        raise ValueError(f"Unsupported mutation format: {mutation}")

    def get_pos(self):
        """ Returns the position of the mutation (0-indexed) """
        return self.position - 1 if self.one_indexed else self.position

    def to_string(self) -> str:
        """Convert back to string format 'A15G'."""
        return f"{self.wt}{self.position}{self.mt}"


class Variant(BaseModel):
    """
    Represents one or more mutations that define a variant sequence.
    Examples:
    - Single: "A15G" -> [Mutation(A, 15, G)]
    - Double: "A15G:K20R" -> [Mutation(A, 15, G), Mutation(K, 20, R)]
    """
    mutations: List[Mutation] = Field(description="List of mutations in this variant")
    wt_sequence: Optional[str] = Field(default=None, description="Reference wild-type sequence")

    @classmethod
    def parse(cls, variant_string: str, one_indexed: bool = True,
              wt_sequence: Optional[str] = None) -> Variant:
        """
        Parse variant string like 'A15G' or 'A15G:K20R'.

        Args:
            variant_string: Mutation string with ':' separator for multiple mutations
            one_indexed: Whether positions are one-indexed
            wt_sequence: Optional WT sequence for validation
        """
        mutation_strings = variant_string.split(':')
        mutations = [Mutation.parse(m, one_indexed=one_indexed) for m in mutation_strings]
        return cls(mutations=mutations, wt_sequence=wt_sequence)

    @field_validator('mutations')
    @classmethod
    def check_no_duplicate_positions(cls, mutations: List[Mutation]) -> List[Mutation]:
        """Ensure no two mutations affect the same position."""
        positions = [m.position for m in mutations]
        if len(positions) != len(set(positions)):
            raise ValueError(f"Duplicate positions in mutations: {positions}")
        return mutations

    def to_string(self) -> str:
        """Convert to string format 'A15G:K20R'."""
        return ':'.join(m.to_string() for m in self.mutations)

    @staticmethod
    def derive_wildtype_sequence(mutation_sequence, variant_string, one_indexed: Optional[bool] = True) -> str:
        variant = Variant.parse(variant_string, one_indexed=one_indexed)
        wt_seq = list(str(mutation_sequence))
        for mutation in variant.mutations:
            if wt_seq[mutation.get_pos()] != mutation.mt:
                raise ValueError(f"Mismatch at position {mutation.position}, did not find the expected mutation")
            wt_seq[mutation.get_pos()] = mutation.wt
        return ''.join(wt_seq)

    def get_mutant_sequence(self, wt_sequence: Optional[str] = None) -> str:
        """
        Create the mutant sequence by applying all mutations to WT.

        Args:
            wt_sequence: Wild-type sequence (uses self.wt_sequence if not provided)
        """
        wt_seq = wt_sequence or self.wt_sequence
        if wt_seq is None:
            raise ValueError("Wild-type sequence required to generate mutant sequence")

        mutant_seq = list(wt_seq)

        for mutation in self.mutations:
            idx = mutation.get_pos()

            # Validate
            if idx >= len(mutant_seq):
                raise ValueError(f"Position {mutation.position} out of range for sequence length {len(wt_seq)}")
            if mutant_seq[idx] != mutation.wt:
                raise ValueError(
                    f"Mismatch at position {mutation.position}: "
                    f"expected '{mutation.wt}', found '{mutant_seq[idx]}'"
                )

            mutant_seq[idx] = mutation.mt

        return ''.join(mutant_seq)

    def mutation_depth(self) -> int:
        return len(self.mutations)

    def __str__(self) -> str:
        return self.to_string()


class SingleMutationScore(BaseModel):
    """Score for a single mutation (can be part of a larger variant)."""
    mutation: Mutation = Field(description="The mutation being scored")
    wt_log_prob: float = Field(description="Log probability of wild-type residue")
    mt_log_prob: float = Field(description="Log probability of mutant residue")

    @computed_field
    @property
    def wt_prob(self) -> float:
        """Probability of wild-type residue."""
        return math.exp(self.wt_log_prob)

    @computed_field
    @property
    def mt_prob(self) -> float:
        """Probability of mutant residue."""
        return math.exp(self.mt_log_prob)

    @computed_field
    @property
    def score(self) -> float:
        """Log-likelihood ratio: log P(mut) - log P(wt)."""
        return self.mt_log_prob - self.wt_log_prob


class VariantScore(BaseModel):
    """
    Score for a mutation variant (single or multiple mutations).
    Can represent different scoring strategies.
    """
    variant: Variant = Field(description="The variant being scored")

    strategy: Optional[ZeroShotMethod] = Field(default=None,
                                               description="Scoring strategy used. "
                                                           "Can be None (e.g. for ground truth / experimental data)")

    mutation_scores: List[SingleMutationScore] = Field(
        description="Individual mutation scores (for decomposition/analysis, can be empty for global scores)"
    )

    total_score: float = Field(
        description="Total variant score (interpretation depends on strategy)"
    )

    model_name: Optional[str] = Field(
        default=None,
        description="Model name used for scoring. Can be None (e.g. for ground truth / experimental data)"
    )

    @classmethod
    def from_marginals(
            cls,
            variant: Variant,
            mutation_scores: List[SingleMutationScore],
            model_name: str,
            method_name: ZeroShotMethod
    ) -> VariantScore:
        """
        Create VariantScore from WT-marginals scoring (additive).
        Total score is sum of individual mutation scores.
        """
        total_score = sum(ms.score for ms in mutation_scores)
        return cls(
            variant=variant,
            strategy=method_name,
            mutation_scores=mutation_scores,
            total_score=total_score,
            model_name=model_name,
        )

    @classmethod
    def from_total_score(
            cls,
            variant: Variant,
            mutation_score: float,
            model_name: str,
            method_name: ZeroShotMethod
    ) -> VariantScore:
        """
        Create VariantScore from pseudoperplexity scoring.
        """
        return cls(
            variant=variant,
            strategy=method_name,
            mutation_scores=[],
            total_score=mutation_score,
            model_name=model_name,
        )

    @classmethod
    def from_experimental(cls,
                          variant: Variant,
                          experimental_score: float):
        """ Create VariantScore from experimental score (might be from a file directly) """
        return cls(variant=variant, total_score=experimental_score, mutation_scores=[])


class RankingResult(BaseModel):
    """ Ranking result for variants of a single protein """
    scc: SerializeAsAny[MetricEstimate] = Field(
        description="Spearmans correlation coefficient (overall ranking quality)")
    ndcg: SerializeAsAny[MetricEstimate] = Field(
        description="Normalized discounted cumulative gain (top-k ranking quality)")

    def scc_score(self):
        """ Rounded SCC mean bootstrapped score """
        return round(self.scc.mean, 3)

    def ndcg_score(self):
        """ Rounded NDCG mean bootstrapped score """
        return round(self.ndcg.mean, 3)

    def __str__(self) -> str:
        return f"Ranking result - SCC: {self.scc_score()}, NDCG: {self.ndcg_score()}"


class AggregatedRankingResult(RankingResult):
    """ Separate class for aggregated ranking results (i.e. multiple assays) to avoid confusion with individual results """

    @classmethod
    def aggregate(cls, results: List[RankingResult]) -> AggregatedRankingResult:
        """
        Aggregate ranking results across multiple assays.
        Follows ProteinGym's bootstrapping approach:
        https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/performance_DMS_benchmarks.py
        """

        from ..functions.bootstrapping import metrics_bootstrap

        # Extract scores (no absolute values)
        scc_scores = [rr.scc_score() for rr in results]
        ndcg_scores = [rr.ndcg_score() for rr in results]

        # Prepare metrics dictionary for bootstrapping
        values = {
            "scc": scc_scores,
            "ndcg": ndcg_scores
        }

        # Bootstrap with default parameters (matching contact dataset approach)
        iterations = 10000  # https://github.com/OATML-Markslab/ProteinGym/blob/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/performance_DMS_benchmarks.py#L95
        bt_res = metrics_bootstrap(
            metrics=values,
            iterations=iterations,
            sample_size=len(results),
            seed=44,
            confidence_level=0.05  # Default confidence level (95% CI)
        )

        # Extract bootstrapped metrics by name
        scc_metric = next(m for m in bt_res if m.name == "scc")
        ndcg_metric = next(m for m in bt_res if m.name == "ndcg")

        # Create aggregated result with bootstrapped confidence intervals
        return AggregatedRankingResult(
            scc=scc_metric,
            ndcg=ndcg_metric,
        )

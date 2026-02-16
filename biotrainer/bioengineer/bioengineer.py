from __future__ import annotations

import torch
import pandas as pd

from pathlib import Path
from typing import List, Optional, Dict, Union

from .bioengineer_models import ESM2Engineer, ProtBertEngineer, ProtGPT2Engineer
from .bioengineer_interfaces import BioEngineerModelWrapper
from .bioengineer_baselines import BioEngineerBaseline, ConstantEngineerBaseline, RandomEngineerBaseline
from .bioengineer_data_classes import VariantScore, ZeroShotMethod, Variant, RankingResult

from ..utilities import get_device
from ..inference import Inferencer
from ..solvers.metrics_calculator import SequenceRegressionMetricsCalculator


class BioEngineer:
    __available_models = [ESM2Engineer, ProtBertEngineer, ProtGPT2Engineer]
    __available_baselines = [ConstantEngineerBaseline, RandomEngineerBaseline]

    def __init__(self, model_wrapper: BioEngineerModelWrapper):
        self.model_wrapper = model_wrapper

    @classmethod
    def from_name(cls, name: str, device: Optional[torch.device] = None) -> BioEngineer:
        device = get_device(device)
        for available_model in (cls.__available_models + cls.__available_baselines):
            model = available_model.detect(name, device=device)
            if model is not None:
                return cls(model)
        raise ValueError(f"No model found for name {name}")

    @classmethod
    def from_baseline(cls, baseline: BioEngineerBaseline) -> BioEngineer:
        for available_baseline in cls.__available_baselines:
            baseline_model = available_baseline.detect(embedder_name=baseline.name, device=torch.device("cpu"))
            if baseline_model is not None:
                return cls(baseline_model)
        raise ValueError(f"No baseline found for name {baseline.name}")

    def zero_shot_wt_marginals(self,
                               wt_sequence: str,
                               mutations: List[str],
                               one_indexed: Optional[bool] = True) -> List[VariantScore]:
        """
        Score mutations using the WT-marginals strategy (no masking).
        The model predicts the logits for all positions at once. Then the marginals of all mutations are calculated.

        Args:
        :param wt_sequence: Wild-type sequence (amino acids).
        :param mutations: List of mutations: Can be single mutations ('A15G')
                or multiple mutations separated by ':' ('A15G:L20P')
        :param one_indexed: Offset for mutation positions (1-indexed by default)

        :return: List of scores or probabilities associated with the specified
            mutations in the sequence.
        :raises:
            NotImplementedError: If logits calculation is not available
        """
        return self.model_wrapper.zero_shot_wt_marginals(wt_sequence, mutations, one_indexed)

    def zero_shot_masked_marginals(self,
                                   wt_sequence: str,
                                   mutations: List[str],
                                   one_indexed: Optional[bool] = True) -> List[VariantScore]:
        """
        Compute zero-shot masked marginals for specific mutations in the given sequence.
        All positions in the sequence are masked sequentially.
        Then the mutation scores are calculated from these marginals.

        :param wt_sequence: Wild-type sequence (amino acids).
        :param mutations: List of mutations: Can be single mutations ('A15G')
                or multiple mutations separated by ':' ('A15G:L20P')
        :param one_indexed: Offset for mutation positions (1-indexed by default)

        :return: List of scores or probabilities associated with the specified
            mutations in the sequence.
        :raises:
            NotImplementedError: If masked logits calculation is not available
        """
        return self.model_wrapper.zero_shot_masked_marginals(wt_sequence, mutations, one_indexed)

    def zero_shot_pseudoperplexity(self,
                                   wt_sequence: str,
                                   mutations: List[str],
                                   one_indexed: Optional[bool] = True,
                                   subtract_wt_pppl: Optional[bool] = True) -> List[VariantScore]:
        """
        Compute the zero-shot pseudoperplexity score for a given sequence and its mutations.

        ⚠️ WARNING: This method is computationally expensive!
        - Requires L forward passes per variant (L = sequence length)
        - For N variants: ~L × N forward passes total
        - Consider using masked-marginals or wt-marginals for large-scale screening

        :param wt_sequence: Wild-type sequence used as a reference for calculating
            pseudoperplexity.
        :param mutations: List of mutations applied to the wild-type sequence. Each
            mutation follows a specific format defined by the implementation.
        :param one_indexed: Determines whether the mutation indices are one-indexed.
            Defaults to True. If False, zero-indexing is assumed.
        :param subtract_wt_pppl: Flag to indicate whether the wild-type pseudoperplexity
            is subtracted from each mutation's pseudoperplexity score. Defaults to True.
        :return: A list of VariantScore objects, each representing the pseudoperplexity
            score associated with a mutation.
        :raises:
            NotImplementedError: If pseudoperplexity calculation is not available
        """
        return self.model_wrapper.zero_shot_pseudoperplexity(wt_sequence, mutations, one_indexed, subtract_wt_pppl)

    def zero_shot_perplexity(self,
                             wt_sequence: str,
                             mutations: List[str],
                             one_indexed: Optional[bool] = True,
                             subtract_wt_ppl: Optional[bool] = True
                             ) -> List[VariantScore]:
        return self.model_wrapper.zero_shot_perplexity(wt_sequence, mutations, one_indexed, subtract_wt_ppl)

    def rank_pgym_dataset(self,
                          dataset_file_path: Union[str, Path],
                          method: ZeroShotMethod,
                          single_mutations_only: bool = False) -> RankingResult:
        """
        Ranks a given ProteinGym dataset using the specified zero-shot method. This method loads the dataset,
        calculates the scores for mutant sequences, and ranks the results against experimentally derived
        fitness scores from the dataset.

        :param dataset_file_path: File path to the ProteinGym dataset. Must be a CSV file containing mutant
                                  sequences and their corresponding experimental fitness scores.
        :param method: Zero-shot prediction method to be used for scoring variant sequences.
        :param single_mutations_only: If True, considers only single mutations in ranking. Defaults to False.

        :return: The ranking result containing the evaluation metrics for predicted mutation scores against
                 the actual ProteinGym scores.

        :raises ValueError: If the specified method is not supported by the model. Additionally raised if
                            the dataset file is empty or missing required columns.
        """
        if method not in self.model_wrapper.supported_methods():
            raise ValueError(f"Method {method} not supported by this model!")

        if isinstance(dataset_file_path, str):
            dataset_file_path = Path(dataset_file_path)

        if not dataset_file_path.exists():
            raise ValueError(f"Dataset file {dataset_file_path} does not exist!")

        # Read ProteinGym dataset
        df = pd.read_csv(dataset_file_path)
        if len(df) == 0:
            raise ValueError(f"Dataset file {dataset_file_path} is empty!")

        try:
            first_row = df.iloc[0]
            mt_seq = first_row["mutated_sequence"]
            mutation_string = first_row["mutant"]
            mutation_fitness = {row["mutant"]: row["DMS_score"] for _, row in df.iterrows()}
            if single_mutations_only:
                mutation_fitness = {mut: score for mut, score in mutation_fitness.items() if ":" not in mut}
            mutations = list(mutation_fitness.keys())
        except KeyError as e:
            raise ValueError(f"Dataset file {dataset_file_path} is missing a required column: {e}")

        # Derive wild-type sequence
        one_indexed = True  # ProteinGym default
        wt_seq = Variant.derive_wildtype_sequence(mutation_sequence=mt_seq, variant_string=mutation_string,
                                                  one_indexed=one_indexed)
        print(f"Wild-type sequence for {dataset_file_path.name} is {wt_seq}")
        print(f"Running {method} on {dataset_file_path.name}...")

        # Calculate variant scores
        subtract_wt_pppl = True  # ProteinGym default for pppl/ppl
        result = None
        match method:
            case ZeroShotMethod.WT_MARGINALS:
                result = self.zero_shot_wt_marginals(wt_sequence=wt_seq, mutations=mutations,
                                                     one_indexed=one_indexed)
            case ZeroShotMethod.MASKED_MARGINALS:
                result = self.zero_shot_masked_marginals(wt_sequence=wt_seq, mutations=mutations,
                                                         one_indexed=one_indexed)
            case ZeroShotMethod.PSEUDOPERPLEXITY:
                result = self.zero_shot_pseudoperplexity(wt_sequence=wt_seq, mutations=mutations,
                                                         one_indexed=one_indexed, subtract_wt_pppl=subtract_wt_pppl)
            case ZeroShotMethod.PERPLEXITY:
                result = self.zero_shot_perplexity(wt_sequence=wt_seq, mutations=mutations, one_indexed=one_indexed,
                                                   subtract_wt_ppl=subtract_wt_pppl)
        assert result is not None, "Zero-shot method returned no results!"

        # Rank variants
        print(f"Ranking results for {dataset_file_path.name}...")
        ranking_result = self.rank_variant_scores(variant_scores=result, actual_scores=mutation_fitness)
        print(f"Ranking result for {dataset_file_path.name}: {ranking_result}")
        return ranking_result

    @staticmethod
    def rank_variant_scores(variant_scores: List[VariantScore], actual_scores: Dict[str, float]) -> RankingResult:
        """ Calculate RankingResult between variant scores and actual scores using bootstrapping.
            The RankingResult includes the typical spearman correlation coefficient as a metric of overall ranking
            performance. Additionally, the NDCG metric is included for evaluation of the top 10% ranking performance.
        """
        variant_dict = {variant_score.variant.to_string(): variant_score.total_score for variant_score in
                        variant_scores}

        if len(variant_dict) != len(actual_scores):
            raise ValueError("Variant scores and actual scores must have the same length!")

        for variant in variant_dict.keys():
            if variant not in actual_scores.keys():
                raise ValueError(f"Variant {variant} not found in actual scores!")

        # Convert dictionaries to tensors
        common_variants = set(variant_dict.keys()) & set(actual_scores.keys())

        v_d = {m: torch.tensor(v) for m, v in variant_dict.items()}
        a_s = {m: torch.tensor(v) for m, v in actual_scores.items()}

        bt_res = Inferencer._do_bootstrapping(iterations=30, sample_size=len(common_variants), confidence_level=0.05,
                                              seq_ids=list(common_variants), all_predictions_dict=v_d,
                                              all_targets_dict=a_s,
                                              metrics_calculator=SequenceRegressionMetricsCalculator(device="cpu",
                                                                                                     n_classes=1)
                                              )
        scc = [res for res in bt_res if res.name == "spearmans-corr-coeff"][0]
        ndcg = [res for res in bt_res if res.name == "ndcg"][0]
        assert scc is not None and ndcg is not None, "Bootstrapping failed!"

        return RankingResult(scc=scc, ndcg=ndcg)

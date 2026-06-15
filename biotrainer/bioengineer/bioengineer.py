from __future__ import annotations

import torch
import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple

from .bioengineer_interfaces import BioEngineerModelWrapper
from .bioengineer_models import ESM2Engineer, ProtBertEngineer, ProtGPT2Engineer
from .bioengineer_custom_model import CustomBioEngineerModel, CustomBioEngineerModelWrapper
from .bioengineer_baselines import BioEngineerBaseline, ConstantEngineerBaseline, RandomEngineerBaseline
from .bioengineer_data_classes import VariantScore, ZeroShotMethod, Variant, RankingResult, ZeroShotContactSingleProtein, ZeroShotContactDatasetResult
from .bioengineer_metrics import evaluate_contact_map

from ..utilities import get_device, is_device_cuda
from ..inference import Inferencer
from ..input_files import read_FASTA
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

    @classmethod
    def from_custom_model(cls, model: Union[CustomBioEngineerModel, BioEngineerModelWrapper],
                          device: Optional[torch.device] = None):
        device = get_device(device)
        if isinstance(model, BioEngineerModelWrapper):
            return cls(model)
        return cls(CustomBioEngineerModelWrapper(custom_bioengineer=model, device=device))

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
                          single_mutations_only: bool = False) -> Tuple[List[VariantScore], RankingResult]:
        """
        Ranks a given ProteinGym dataset using the specified zero-shot method. This method loads the dataset,
        calculates the scores for mutant sequences, and ranks the results against experimentally derived
        fitness scores from the dataset.

        :param dataset_file_path: File path to the ProteinGym dataset. Must be a CSV file containing mutant
                                  sequences and their corresponding experimental fitness scores.
        :param method: Zero-shot prediction method to be used for scoring variant sequences.
        :param single_mutations_only: If True, considers only single mutations in ranking. Defaults to False.

        :return: Tuple containing: [0] The list of calculated variant scores,
                [1] The ranking result containing the evaluation metrics for predicted mutation scores against
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
        return result, ranking_result

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

    # ============================================================================
    # Zero-shot contact task entry point
    # ============================================================================

    def zero_shot_contact_map(self, sequence: str, batch_size: int = 32) -> np.ndarray:
        """
        Derive contact map from categorical Jacobian.

        Returns:
            np.ndarray: [L, L]
        """
        return self.model_wrapper.zero_shot_contact_map(sequence, batch_size)

    def run_contact_dataset(self,
                          dataset_name: str,
                          fasta_file_path: Union[str, Path],
                          contacts_dir_path: Union[str, Path],
                          method: ZeroShotMethod) -> Tuple[List[ZeroShotContactSingleProtein], ZeroShotContactDatasetResult]:
        """
        Given a dataset, computes and evaluates contact maps for all proteins in the dataset, using the categorical jacobian based zero-shot method. 
        This function loads the dataset, including the ground truth contact maps, and evaluates the predicted contact map per protein.
        The results (precision scores for topk predicted contacts) are aggregated over the dataset.

        Args:
            dataset_name: Name of the dataset.
            fasta_file_path: Path to the fasta file containing the protein sequences.
            contacts_dir_path: Path to the directory containing the ground truth contact maps.
                               (expected naming format per protein: <protein_ID>.npy)
            method: Zero-shot prediction method to be used for computing contact maps (JACOBIAN_CONTACT).

        Returns:
            Tuple[List[ZeroShotContactSingleProtein], ZeroShotContactDatasetResult]: List of per protein results and aggregated dataset result.

        Raises:
            ValueError: If the specified method is not supported by the wrapped model; or if the dataset files are empty or missing.
        """
        if method not in self.model_wrapper.supported_methods():
            raise ValueError(f"Method {method} not supported by this model!")

        if isinstance(fasta_file_path, str):
            fasta_file_path = Path(fasta_file_path)

        if not fasta_file_path.exists():
            raise ValueError(f"Fasta file {fasta_file_path} does not exist!")

        if isinstance(contacts_dir_path, str):
            contacts_dir_path = Path(contacts_dir_path)

        if not contacts_dir_path.exists():
            raise ValueError(f"Contacts directory {contacts_dir_path} does not exist!")

        # Read fasta file
        seq_records = read_FASTA(fasta_file_path)
        if len(seq_records) == 0:
            raise ValueError(f"Fasta file {fasta_file_path} is empty!")

        per_protein_results: List[ZeroShotContactSingleProtein] = [] 
        for record in seq_records:
            seq_id = record.seq_id
            sequence = record.seq

            ground_truth_contact_map_path = contacts_dir_path / f"{seq_id}.npy"
            if not ground_truth_contact_map_path.exists():
                raise ValueError(f"Contact map {ground_truth_contact_map_path} does not exist!")
            ground_truth_contact_map = np.load(ground_truth_contact_map_path)
            if ground_truth_contact_map.size == 0:
                raise ValueError(f"Empty contact map for {seq_id}")
            if ground_truth_contact_map.shape != (len(sequence), len(sequence)):
                raise ValueError(f"Shape mismatch for {seq_id}: expected ({len(sequence)}, {len(sequence)}), got {ground_truth_contact_map.shape}")

            predicted_contact_map = None
            match method:
                case ZeroShotMethod.JACOBIAN_CONTACT:
                    predicted_contact_map = self.zero_shot_contact_map(sequence) #TODO: let user specify batch size; for now, default=32!
            assert predicted_contact_map is not None, "Zero-shot method returned no contact map!"

            precision_scores = evaluate_contact_map(predicted_contact_map, ground_truth_contact_map)
            single_protein_result = ZeroShotContactSingleProtein(protein_name=seq_id, precision_scores=precision_scores)
            per_protein_results.append(single_protein_result)
            # TODO: review if explicit freeing up required or not in between proteins
            # if is_device_cuda(self.model_wrapper._device):
            #     torch.cuda.empty_cache()

        assert len(per_protein_results) > 0, "Empty per-protein results!"

        # Aggregate results
        print(f"Aggregating results for {dataset_name}...")
        dataset_result = ZeroShotContactDatasetResult.aggregate_results(dataset_name=dataset_name, per_protein_results=per_protein_results)
        print(f"Result for {dataset_name}: {dataset_result}")
        return per_protein_results, dataset_result
import torch
import pickle

from pathlib import Path
from typing import Dict, Optional, Any

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .constants import MASK_AND_LABELS_PAD_VALUE

from ..protocols import Protocol


class FeatureScaler:

    def __init__(self, method: str, protocol: Protocol, loaded_scaler: Optional[Any] = None):
        self.method = method
        self.protocol = protocol
        self.scaler = loaded_scaler

    @classmethod
    def load(cls, method: str, protocol: Protocol, load_path: Path):
        with open(load_path, "rb") as f:
            loaded_scaler = pickle.load(f)
        return cls(method=method, protocol=protocol, loaded_scaler=loaded_scaler)

    @staticmethod
    def methods():
        return {"none": None,
                "standard": StandardScaler(),
                "minmax": MinMaxScaler()}

    def _fit_per_sequence(self, x: Dict[str, torch.Tensor]):
        # Stack all embeddings and fit
        embeddings = torch.stack(list(x.values())).numpy()
        self.scaler.fit(embeddings)

    def _fit_per_residue(self, x: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        # Collect all residue embeddings, excluding masked positions
        all_residue_embeddings = []

        for seq_id, emb in x.items():
            # emb shape: (seq_len, embedding_dim)
            assert targets is not None and seq_id in targets, (f"No target info available for sequence {seq_id} "
                                                               f"during scaling!")
            target = targets[seq_id]
            # Create mask: True for positions to INCLUDE (not masked)
            mask = target != MASK_AND_LABELS_PAD_VALUE
            # Apply mask to select only unmasked residues
            unmasked_embeddings = emb[mask]  # shape: (num_unmasked, embedding_dim)
            all_residue_embeddings.append(unmasked_embeddings)


        # Concatenate all residue embeddings: (total_residues, embedding_dim)
        concatenated_embeddings = torch.cat(all_residue_embeddings, dim=0).numpy()

        # Fit scaler on all residues
        self.scaler.fit(concatenated_embeddings)

    def fit(self, x: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """
        Fit the scaler on training embeddings.

        Args:
            x: Dictionary mapping sequence IDs to embeddings
               - Per-sequence: shape (embedding_dim,)
               - Per-residue: shape (seq_len, embedding_dim)
            targets: Dictionary of targets for identifying masked positions
                     (only used for per-residue tasks to exclude masked residues)
        """
        self.scaler = self.methods().get(self.method, None)
        assert self.scaler is not None, f"Feature scaler called with unknown method: {self.method}"

        if self.protocol in Protocol.using_per_sequence_embeddings():
            self._fit_per_sequence(x)

        else:  # per-residue embeddings
            self._fit_per_residue(x, targets)

        return self

    def _transform_per_sequence(self, x: Dict[str, torch.tensor]):
        embeddings = torch.stack(list(x.values())).numpy()
        scaled_embeddings = self.scaler.transform(embeddings)
        return {seq_id: torch.tensor(scaled_emb) for seq_id, scaled_emb in zip(x.keys(), scaled_embeddings)}

    def _transform_per_residue(self, x: Dict[str, torch.tensor]):
        scaled_dict = {}

        for seq_id, emb in x.items():
            # emb shape: (seq_len, embedding_dim)
            original_shape = emb.shape

            # Transform: reshape to 2D, apply scaler, reshape back
            emb_np = emb.numpy()
            scaled_emb_np = self.scaler.transform(emb_np)  # Already 2D, no reshaping needed
            scaled_emb = torch.tensor(scaled_emb_np, dtype=emb.dtype)

            assert scaled_emb.shape == original_shape, \
                f"Shape mismatch after scaling: {scaled_emb.shape} != {original_shape}"

            scaled_dict[seq_id] = scaled_emb

        return scaled_dict

    def transform(self, x: Dict[str, torch.tensor]):
        assert self.scaler is not None, f"Feature scaler called without fitting!"

        if self.protocol in Protocol.using_per_sequence_embeddings():
            return self._transform_per_sequence(x)
        else: # per-residue embeddings
            return self._transform_per_residue(x)

    def save(self, save_path: Path):
        with open(save_path, "wb") as f:
            pickle.dump(self.scaler, f)

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Callable

from biotrainer_core.data_classes import SequenceData


# Not using pydantic for faster transfer
@dataclass(frozen=True)
class EmbeddingDatabaseDTO:
    hash_key: str
    seq_len: int
    access_count: int
    created_at: datetime
    last_accessed: datetime
    embedder_name: str
    embd_per_sequence: Optional  # Can be numpy array or torch tensor or bytes (compressed)
    embd_per_residue: Optional  # Can be numpy array or torch tensor or bytes (compressed)
    keep: bool  # Keep this embedding in the database and do not remove it via cleanup

    @classmethod
    def from_sequence_data(cls, sequence_data: SequenceData, embedder_name: str, reduce: bool):
        return EmbeddingDatabaseDTO(
            hash_key=sequence_data.get_hash(),
            seq_len=len(sequence_data.seq),
            access_count=0,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            embedder_name=embedder_name,
            embd_per_sequence=sequence_data.embedding if reduce else None,
            embd_per_residue=sequence_data.embedding if not reduce else None,
            keep=True,
        )

    def reduced(self) -> bool:
        return self.embd_per_sequence is not None and self.embd_per_residue is None

    def to_tuple(self) -> Tuple[str, int, int, datetime, datetime, str, Optional, Optional, bool]:
        return (self.hash_key,
                self.seq_len,
                self.access_count,
                self.created_at,
                self.last_accessed,
                self.embedder_name,
                self.embd_per_sequence,
                self.embd_per_residue,
                self.keep)

    def database_key(self) -> str:
        return f"{self.hash_key}_{self.cleaned_embedder_name()}"

    @staticmethod
    def decompose_database_key(db_key: str) -> Tuple[str, str]:
        vals = db_key.split("_")
        hash_key = vals[0]
        embedder_name = vals[1] + "/" + "_".join(vals[2:]) if len(vals) > 2 else vals[1]
        return hash_key, embedder_name

    def cleaned_embedder_name(self) -> str:
        return self.embedder_name.replace('/', '_')

    def compressed(self, compression_strategy: Callable) -> EmbeddingDatabaseDTO:
        per_seq_compressed = compression_strategy(self.embd_per_sequence)
        per_res_compressed = compression_strategy(self.embd_per_residue)
        return EmbeddingDatabaseDTO(
            hash_key=self.hash_key,
            seq_len=self.seq_len,
            access_count=self.access_count,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            embedder_name=self.embedder_name,
            embd_per_sequence=per_seq_compressed,
            embd_per_residue=per_res_compressed,
            keep=self.keep,
        )

    def decompressed(self, decompression_strategy: Callable) -> EmbeddingDatabaseDTO:
        per_seq_decompressed = decompression_strategy(self.embd_per_sequence)
        per_res_decompressed = decompression_strategy(self.embd_per_residue)
        return EmbeddingDatabaseDTO(
            hash_key=self.hash_key,
            seq_len=self.seq_len,
            access_count=self.access_count,
            created_at=self.created_at,
            last_accessed=self.last_accessed,
            embedder_name=self.embedder_name,
            embd_per_sequence=per_seq_decompressed,
            embd_per_residue=per_res_decompressed,
            keep=self.keep,
        )

    def copy_with(
            self,
            hash_key: Optional[str] = None,
            seq_len: Optional[int] = None,
            access_count: Optional[int] = None,
            created_at: Optional[datetime] = None,
            last_accessed: Optional[datetime] = None,
            embedder_name: Optional[str] = None,
            embd_per_sequence: Optional = None,
            embd_per_residue: Optional = None,
            keep: Optional[bool] = None,
    ) -> EmbeddingDatabaseDTO:
        return EmbeddingDatabaseDTO(
            hash_key=hash_key if hash_key is not None else self.hash_key,
            seq_len=seq_len if seq_len is not None else self.seq_len,
            access_count=access_count if access_count is not None else self.access_count,
            created_at=created_at if created_at is not None else self.created_at,
            last_accessed=last_accessed if last_accessed is not None else self.last_accessed,
            embedder_name=embedder_name if embedder_name is not None else self.embedder_name,
            embd_per_sequence=embd_per_sequence if embd_per_sequence is not None else self.embd_per_sequence,
            embd_per_residue=embd_per_residue if embd_per_residue is not None else self.embd_per_residue,
            keep=keep if keep is not None else self.keep,
        )
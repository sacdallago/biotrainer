# Built with ESM (repository: github.com/facebookresearch/esm)
import torch

from typing import List, Any, Generator
from transformers import EsmTokenizer, EsmModel

from .huggingface_transformer_embedder import HuggingfaceTransformerEmbedder

from ..interfaces import BioOptEmbedder, preprocess_sequences_without_whitespaces

from ...utilities import get_logger

logger = get_logger(__name__)

_esm2_family_dict = {'facebook/esm2_t6_8M_UR50D': 320,
                     'facebook/esm2_t12_35M_UR50D': 480,
                     'facebook/esm2_t30_150M_UR50D': 640,
                     'facebook/esm2_t33_650M_UR50D': 1280,
                     'facebook/esm2_t36_3B_UR50D': 2560,
                     'facebook/esm2_t48_15B_UR50D': 5120,
                     }


class ESM2(HuggingfaceTransformerEmbedder, BioOptEmbedder):

    @classmethod
    def detect(cls, embedder_name: str, use_half_precision: bool, dtype: torch.dtype, device: torch.device):
        esm2_family = _esm2_family_dict.keys()
        if embedder_name in esm2_family:
            # Load the tokenizer
            tokenizer = EsmTokenizer.from_pretrained(embedder_name, do_lower_case=False, dtype=dtype)
            # Load the model
            model = EsmModel.from_pretrained(embedder_name, dtype=dtype).to(device)
            return cls(name=embedder_name, model=model, tokenizer=tokenizer, use_half_precision=use_half_precision,
                       device=device)
        return None

    def embedding_dim(self) -> int:
        return _esm2_family_dict[self.name]

    def _find_preprocessing_strategy(self):
        strategy = preprocess_sequences_without_whitespaces
        logger.info(f"Chosen sequence pre-processing strategy: {strategy.__name__}")
        return strategy

    def _embed_batch_implementation(self, batch: List[str], model: Any) -> Generator[
        torch.Tensor, None, None]:
        """
        Optimized ESM2 implementation using attention masks for post-processing.

        ESM2 sequences after tokenization typically look like:
        [BOS, AA1, AA2, AA3, ..., AAn, EOS, PAD, PAD, ...]

        We want to return only the residue embeddings [AA1, AA2, ..., AAn]
        without costly scanning for special tokens each time.
        """
        tokenized_sequences, attention_mask = self._tokenize(batch)

        with self._get_gradient_context():
            embeddings = model(
                input_ids=tokenized_sequences,
                attention_mask=attention_mask,
            )

        embeddings = embeddings.last_hidden_state

        # Process all sequences before yielding (keeps them on GPU longer)
        processed_embeddings = []
        for seq_num in range(len(embeddings)):
            # Count non-padding tokens using attention mask (includes BOS and EOS)
            num_real_tokens = attention_mask[seq_num].sum().item()

            # Extract embeddings: skip BOS (first token) and EOS (last real token) and all padding
            # From [BOS, AA1, AA2, ..., AAn, EOS, PAD, PAD] -> [AA1, AA2, ..., AAn]
            embedding = embeddings[seq_num, 1:num_real_tokens - 1]
            processed_embeddings.append(embedding)

        # Yield all at once
        yield from processed_embeddings

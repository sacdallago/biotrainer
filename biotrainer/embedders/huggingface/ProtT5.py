import torch

from typing import List, Any, Generator
from transformers import T5Tokenizer, T5EncoderModel

from .huggingface_transformer_embedder import HuggingfaceTransformerEmbedder

from ..interfaces import BioOptEmbedder, preprocess_sequences_with_whitespaces

from ...utilities import get_logger

logger = get_logger(__name__)


class ProtT5(HuggingfaceTransformerEmbedder, BioOptEmbedder):
    @classmethod
    def detect(cls, embedder_name: str, use_half_precision: bool, dtype: torch.dtype, device: torch.device):
        prott5_names = ['Rostlab/prot_t5_xl_half_uniref50-enc', 'Rostlab/prot_t5_xl_uniref50']
        if embedder_name in prott5_names:
            # Load the tokenizer
            tokenizer = T5Tokenizer.from_pretrained(embedder_name, do_lower_case=False, dtype=dtype)
            # Load the model
            model = T5EncoderModel.from_pretrained(embedder_name, dtype=dtype).to(device)
            return cls(name=embedder_name, model=model, tokenizer=tokenizer, use_half_precision=use_half_precision,
                       device=device)
        return None

    def embedding_dim(self) -> int:
        return 1024

    def _find_preprocessing_strategy(self):
        strategy = preprocess_sequences_with_whitespaces
        logger.info(f"Chosen sequence pre-processing strategy: {strategy.__name__}")
        return strategy

    def _embed_batch_implementation(self, batch: List[str], model: Any) -> Generator[
        torch.Tensor, None, None]:
        """
        Optimized ProtT5 implementation using attention masks for post-processing.

        ProtT5 sequences after tokenization look like:
        [AA1, AA2, AA3, ..., AAn, EOS, PAD, PAD, ...]

        The embedding is only [AA1, AA2, AA3, ..., AAn] (no EOS, no padding).
        """
        tokenized_sequences, attention_mask = self._tokenize(batch)

        with self._get_gradient_context():
            embeddings = model(
                input_ids=tokenized_sequences,
                attention_mask=attention_mask,
            )

        embeddings = embeddings.last_hidden_state  # or embeddings[0]

        # Process all sequences before yielding (keeps them on GPU longer)
        processed_embeddings = []
        for seq_num in range(len(embeddings)):
            # Count non-padding tokens using attention mask (includes EOS)
            num_real_tokens = attention_mask[seq_num].sum().item()

            # Extract embeddings: skip EOS (last real token) and all padding
            # This gets: [AA1, AA2, ..., AAn] from [AA1, AA2, ..., AAn, EOS, PAD, PAD]
            embedding = embeddings[seq_num, :num_real_tokens - 1]
            processed_embeddings.append(embedding)

        # Yield all at once
        yield from processed_embeddings

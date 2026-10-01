import torch

from typing import List, Any, Generator
from transformers import BertModel, BertTokenizer

from .huggingface_transformer_embedder import HuggingfaceTransformerEmbedder

from ..interfaces import BioOptEmbedder, preprocess_sequences_with_whitespaces

from ...shared import get_logger

logger = get_logger(__name__)


class ProtBert(HuggingfaceTransformerEmbedder, BioOptEmbedder):
    @classmethod
    def detect(cls, embedder_name: str, use_half_precision: bool, dtype: torch.dtype, device: torch.device):
        if embedder_name == "Rostlab/prot_bert":
            # Load the tokenizer
            tokenizer = BertTokenizer.from_pretrained(embedder_name, do_lower_case=False, dtype=dtype)
            # Load the model
            model = BertModel.from_pretrained(embedder_name, dtype=dtype).to(device)
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
        Optimized ProtBert implementation using attention masks for post-processing.

        ProtBert sequences after tokenization typically look like:
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

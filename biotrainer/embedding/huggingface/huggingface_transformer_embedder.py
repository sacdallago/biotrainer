# Inspired by bio_embeddings embed module (https://github.com/sacdallago/bio_embeddings/tree/develop/bio_embeddings/embed)
import torch

from peft import PeftModel
from typing import List, Generator, Any, Union, Optional

from torch import Tensor

from ..interfaces import EmbedderWithFallback


from ...shared import get_logger

logger = get_logger(__name__)


class HuggingfaceTransformerEmbedder(EmbedderWithFallback):
    """ Generic Huggingface Transformer Embedder"""

    def __init__(self, name: str, model, tokenizer, use_half_precision: bool, device: Union[str, torch.device]):
        self.name = name
        self._model = model
        self._tokenizer = tokenizer
        self._use_half_precision = use_half_precision
        self._device = device
        self._custom_indices_to_remove = self._get_custom_indices_to_remove()
        self._mask_token_id: Optional[int] = self.get_mask_token_id()
        self._set_model_precision()

    def _set_model_precision(self):
        if self._use_half_precision and self._device == "cpu":
            # This is caught earlier, but we check it here again for safety
            raise NotImplementedError("Cannot use half_precision mode together with cpu!")
        if self._use_half_precision:
            try:
                self._model = self._model.half()
            except AttributeError:
                raise NotImplementedError(f"Given model {self.name} does not support half_precision mode!")

    def _get_gradient_context(self):
        if isinstance(self._model, PeftModel) and self._model.training:
            return torch.enable_grad()  # Finetuning
        return torch.no_grad()  # Usual embeddings inference

    def _get_fallback_model(self):
        """ Returns the CPU model """
        if self._use_half_precision:
            raise NotImplementedError(
                "You sequence was too long for the GPU, "
                "but we can't fall back to the CPU with half_precision_model=True "
                "(https://github.com/huggingface/transformers/issues/11546)"
            )
        return self._model.to("cpu")

    def _ensure_attention_backend(self):
        """
        Ensure that the model uses an attention implementation that can return attention maps.

        Some optimized attention implementations do not support output_attentions=True.
        The eager backend is slower, so this is only enabled lazily for attention-map extraction.
        """
        config = getattr(self._model, "config", None)
        current_backend = getattr(config, "_attn_implementation", None)

        if current_backend == "eager":
            return

        if hasattr(self._model, "set_attn_implementation"):
            self._model.set_attn_implementation("eager")
            return

        if config is not None and hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"
            return

        raise NotImplementedError(
            f"Model {self.name} does not expose a supported way to switch to the eager attention backend. "
            "Load this model with attn_implementation='eager' if attention maps are required."
        )

    def _embed_single(self, sequence: str) -> torch.Tensor:
        [embedding] = self._embed_batch([sequence])
        return embedding

    def _remove_special_tokens(self, embedding: torch.Tensor, input_id: torch.Tensor) -> torch.Tensor:
        """
        Remove special tokens from the embedding.

        :param embedding: The per-residue embedding for a single sequence
        :param input_id: The input ids for the sequence
        :return: The embedding with special token indices removed
        """
        special_tokens_mask = self._tokenizer.get_special_tokens_mask(input_id, already_has_special_tokens=True)
        # Replace all special tokens but the mask token for MLM
        indices_to_remove = [index for index, mask in enumerate(special_tokens_mask)
                             if mask != 0 and input_id[index] != self._mask_token_id]
        indices_to_remove += self._custom_indices_to_remove
        indices_to_remove = list(set(indices_to_remove))

        # Create a boolean mask for indices to keep
        keep_mask = torch.ones(embedding.size(0), dtype=torch.bool, device=embedding.device)
        keep_mask[indices_to_remove] = False

        return embedding[keep_mask]

    def _embed_batch_implementation(self, batch: List[str], model: Any) -> Generator[
        torch.Tensor, None, None]:
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
            input_id = tokenized_sequences[seq_num]
            embedding = self._remove_special_tokens(embeddings[seq_num], input_id)
            processed_embeddings.append(embedding)

        # Yield all at once
        yield from processed_embeddings

    # From https://github.com/facebookresearch/esm/blob/main/esm/modules.py
    # and https://github.com/chandar-lab/AMPLIFY/blob/main/examples/utils.py
    @staticmethod
    def _apc(x: torch.Tensor):
        """Perform average product correct, used for contact prediction."""
        a1 = x.sum(-1, keepdims=True)
        a2 = x.sum(-2, keepdims=True)
        a12 = x.sum((-1, -2), keepdims=True)

        avg = a1 * a2
        avg.div_(a12)  # in-place to reduce memory
        normalized = x - avg
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    # From https://github.com/facebookresearch/esm/blob/main/esm/modules.py
    # and https://github.com/chandar-lab/AMPLIFY/blob/main/examples/utils.py
    @staticmethod
    def _symmetrize(x: torch.Tensor):
        """ Make layer symmetric in final two dimensions, used for contact prediction. """
        return x + x.transpose(-1, -2)

    # Adapted from https://github.com/chandar-lab/AMPLIFY/blob/main/examples/contact_prediction.ipynb
    def compute_attention_map(self, sequence: str) -> torch.Tensor:
        self._ensure_attention_backend()
        with torch.no_grad(), torch.autocast(device_type=str(self._device), dtype=torch.float16, enabled=True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            x = torch.as_tensor(self._tokenizer.encode(sequence)).to(torch.long)  # tokenize the protein
            x = x.unsqueeze(0).to(self._device)
            attn_map = self._model(x, output_attentions=True)["attentions"]
            attn_map = torch.stack(attn_map).detach().cpu()  # stack the attention maps and move to CPU
            attn_map = attn_map.reshape(-1, x.size(-1), x.size(-1))  # (map, residues, residues)
            attn_map = attn_map[:, 1:-1, 1:-1]  # remove special tokens <bos> and <eos>
            attn_map = self._apc(self._symmetrize(attn_map))  # process the attention maps
            attn_map = attn_map.permute(1, 2, 0)  # (residues, residues, map)
            return attn_map

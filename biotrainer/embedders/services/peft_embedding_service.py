from pathlib import Path
from typing import Dict, Any

from ..interfaces import EmbedderInterface
from .embedding_service import EmbeddingService

from ...utilities import get_logger

logger = get_logger(__name__)


class PeftEmbeddingService(EmbeddingService):
    _lora_applied = False

    def __init__(self, embedder: EmbedderInterface = None, use_half_precision: bool = False,
                 finetuning_config: Dict[str, Any] = None):
        super().__init__(embedder, use_half_precision)
        self._finetuning_config = finetuning_config
        self._lora_applied = self._apply_lora()

    def save_embedder(self, output_dir: Path):
        logger.info(f"Saving fine-tuned embedder to {output_dir}")
        assert self._lora_applied, f"LoRA adapter was not applied before saving it!"

        from peft import PeftModel
        assert isinstance(self._embedder._model, PeftModel), f"{self.__class__.__name__} is not a PeftModel!"

        self._embedder._model.save_pretrained(str(output_dir))

    def _apply_lora(self):
        """Apply LoRA adapters to the underlying model"""
        assert not self._lora_applied, "Tried to apply lora adapter multiple times for the same model!"

        from peft import LoraConfig, get_peft_model

        # Extract LoRA parameters from config
        target_modules = self._finetuning_config.get("lora_target_modules", "auto")
        if target_modules == "auto":
            target_modules = self._get_default_target_modules_by_embedder()

        lora_config = LoraConfig(
            r=self._finetuning_config.get("lora_r", 8),
            lora_alpha=self._finetuning_config.get("lora_alpha", 16),
            target_modules=target_modules,
            lora_dropout=self._finetuning_config.get("lora_dropout", 0.05),
            bias=self._finetuning_config.get("lora_bias", "none"),
            inference_mode=False,
        )

        # Apply LoRA adapters to the embedder model
        model = self._embedder._model
        if model is None:
            raise ValueError(f"{self._embedder.name} does not provide a model for finetuning!")

        peft_model = get_peft_model(model=model, peft_config=lora_config)
        peft_model = peft_model.train()

        self._embedder._model = peft_model

        logger.info("Successfully applied lora config to embedder model! Trainable parameters:")
        self._embedder._model.print_trainable_parameters()

        return True

    def _get_default_target_modules_by_embedder(self):
        embedder_name = self._embedder.name.lower()
        if "t5" in embedder_name:
            return ["q", "k", "v", "o"]
        elif "esm" in embedder_name:
            return ["query", "key", "value"]
        elif "bert" in embedder_name:
            return ["query", "key", "value", "dense"]
        return ["query", "key", "value"]

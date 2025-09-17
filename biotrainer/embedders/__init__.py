import torch

from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any
from transformers import AutoTokenizer, T5Tokenizer, T5EncoderModel, EsmTokenizer, EsmModel, BertTokenizer, \
    BertForMaskedLM

from .onnx import OnnxEmbedder
from .huggingface import HuggingfaceTransformerEmbedder
from .interfaces import EmbedderInterface, CustomTokenizer
from .services import EmbeddingService, PeftEmbeddingService
from .predefined_embedders import RandomEmbedder, AAOntologyEmbedder, OneHotEncodingEmbedder, Blosum62Embedder

from ..utilities import is_device_cpu, get_logger

__PREDEFINED_EMBEDDERS = {
    "one_hot_encoding": OneHotEncodingEmbedder,
    "random_embedder": RandomEmbedder,
    "AAOntology": AAOntologyEmbedder,
    "blosum62": Blosum62Embedder,
}

logger = get_logger(__name__)


def get_embedding_service(embedder_name: str,
                          custom_tokenizer_config: Optional[str],
                          use_half_precision: Optional[bool] = False,
                          device: Optional[Union[str, torch.device]] = None,
                          finetuning_config: Optional[Dict[str, Any]] = None) -> Union[
    EmbeddingService, PeftEmbeddingService]:
    embedder: EmbedderInterface = _get_embedder(embedder_name=embedder_name,
                                                custom_tokenizer_config=custom_tokenizer_config,
                                                use_half_precision=use_half_precision,
                                                device=device)
    if finetuning_config:
        return PeftEmbeddingService(embedder=embedder, use_half_precision=use_half_precision,
                                    finetuning_config=finetuning_config)

    return EmbeddingService(embedder=embedder, use_half_precision=use_half_precision)


def _determine_tokenizer_and_model(embedder_name: str) -> Tuple:
    """
    Method to find the model and tokenizer architecture from the embedder name for huggingface transformers.
    AutoTokenizer does not work for all models, such that the next best option is to use the embedder name.

    @param embedder_name: Name of huggingface transformer
    @return: Tuple of tokenizer and model class
    """
    try:
        auto_tok = AutoTokenizer.from_pretrained(embedder_name)
        embedder_class = auto_tok.__class__.__name__
    except Exception:
        embedder_class = embedder_name

    if "t5" in embedder_class.lower():
        return T5Tokenizer, T5EncoderModel
    elif "esm" in embedder_class.lower():
        return EsmTokenizer, EsmModel
    elif "bert" in embedder_class.lower():
        return BertTokenizer, BertForMaskedLM

    # Use T5 as default
    return T5Tokenizer, T5EncoderModel


def _get_embedder(embedder_name: Optional[str],
                  custom_tokenizer_config: Optional[str],
                  use_half_precision: Optional[bool],
                  device: Optional[Union[str, torch.device]]) -> EmbedderInterface:
    if embedder_name is None or embedder_name == "":
        raise ValueError("No embedder name provided!")

    # Predefined Embedders
    if embedder_name in __PREDEFINED_EMBEDDERS.keys():
        logger.info(f"Using predefined embedder: {embedder_name}")
        return __PREDEFINED_EMBEDDERS[embedder_name]()

    # Custom Embedders
    if embedder_name.endswith(".py"):  # Check that file ends on .py => Python script
        raise ValueError(f"Custom embedders from script have been replaced with onnx models in v0.9.7!")

    # ONNX Embedders
    if embedder_name.endswith(".onnx"):
        if custom_tokenizer_config:
            tokenizer = CustomTokenizer.from_config(config_path=Path(custom_tokenizer_config))
            logger.info(f"Using custom tokenizer (vocab size: {tokenizer.vocab_size})!")
        else:
            tokenizer = T5Tokenizer.from_pretrained(embedder_name, dtype=torch.float32)
        onnx_model = OnnxEmbedder(onnx_path=Path(embedder_name), tokenizer=tokenizer, device=device)
        logger.info(f"Using onnx embedder: {onnx_model.name}")
        return onnx_model

    # Huggingface Transformer Embedders
    if use_half_precision and is_device_cpu(device):
        raise ValueError(f"use_half_precision mode is not compatible with embedding "
                         f"on the CPU. (See: https://github.com/huggingface/transformers/issues/11546)")

    dtype = torch.float16 if use_half_precision else torch.float32

    tokenizer_class, model_class = _determine_tokenizer_and_model(embedder_name)
    logger.info(f"Loading embedder model {embedder_name}..")
    try:
        tokenizer = tokenizer_class.from_pretrained(embedder_name, dtype=dtype)
        model = model_class.from_pretrained(embedder_name, dtype=dtype)
    except OSError as os_error:
        raise Exception(f"{embedder_name} could not be found!") from os_error
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(embedder_name, dtype=dtype)
            model = T5EncoderModel.from_pretrained(embedder_name, dtype=dtype)
        except Exception as e:
            raise Exception(f"Loading {embedder_name} automatically and as {tokenizer_class.__class__.__name__} failed!"
                            f" Please provide a custom_embedder script for your use-case.") from e

    logger.info(f"Using huggingface transformer embedder: {embedder_name} "
                f"- Model: {model.__class__.__name__} "
                f"- Tokenizer: {tokenizer.__class__.__name__} "
                f"- Half-Precision: {str(use_half_precision)}")
    model.to(device)
    return HuggingfaceTransformerEmbedder(name=embedder_name, model=model, tokenizer=tokenizer,
                                          use_half_precision=use_half_precision, device=device)


def get_predefined_embedder_names() -> List[str]:
    return list(__PREDEFINED_EMBEDDERS.keys())


__all__ = [
    "EmbeddingService",
    "PeftEmbeddingService",
    "OneHotEncodingEmbedder",
    "RandomEmbedder",
    "AAOntologyEmbedder",
    "get_embedding_service",
    "get_predefined_embedder_names"
]


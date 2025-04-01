from pathlib import Path

import onnx
import torch
import onnxruntime
import numpy as np

from numpy import ndarray
from typing import Union, List, Any, Generator
from transformers import PreTrainedTokenizer

from .custom_tokenizer import CustomTokenizer
from .embedder_interfaces import EmbedderWithFallback

from ..utilities import is_device_cuda


class OrtSessionWrapper:
    def __init__(self, session):
        self.session = session

    def to(self, device):
        return self

    def run(self, output_names, input_feed, run_options=None):
        return self.session.run(output_names, input_feed=input_feed, run_options=run_options)


class OnnxEmbedder(EmbedderWithFallback):

    def __init__(self, onnx_path: Path, tokenizer: PreTrainedTokenizer, device: Union[str, torch.device]):
        self.name = f"onnx-{onnx_path.name.replace('.onnx', '')}"
        self._onnx_path = onnx_path
        self._device = device
        self._model = self._load_model()

        self._tokenizer = tokenizer
        if isinstance(tokenizer, CustomTokenizer):
            self._preprocessing_strategy = tokenizer.preprocessing_strategy

    @staticmethod
    def check_onnxruntime_gpu():
        # TODO [Cross platform] Add support for mps/coreML (needs onnxruntime-coreml)
        if torch.cuda.is_available():
            if 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
                print("CUDA is available but onnxruntime-gpu is not installed. "
                      "Install it with: 1. poetry remove onnxruntime 2. poetry add onnxruntime-gpu")

    def _load_model(self) -> OrtSessionWrapper:
        # Create ONNX Runtime session
        onnx_model = onnx.load(self._onnx_path)
        onnx.checker.check_model(onnx_model)

        # TODO [Cross platform] Add support for mps/coreML (needs onnxruntime-coreml)
        if is_device_cuda(self._device):
            ep_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            ep_list = ['CPUExecutionProvider']

        ort_session = onnxruntime.InferenceSession(self._onnx_path, providers=ep_list)

        return OrtSessionWrapper(ort_session)

    def _get_fallback_model(self) -> OrtSessionWrapper:
        onnx_model = onnx.load(self._onnx_path)
        onnx.checker.check_model(onnx_model)

        ep_list = ['CPUExecutionProvider']

        ort_session = onnxruntime.InferenceSession(self._onnx_path, providers=ep_list)
        return OrtSessionWrapper(ort_session)

    def _embed_single(self, sequence: str) -> ndarray:
        [embedding] = self._embed_batch([sequence])
        return embedding

    def _embed_batch_implementation(self, batch: List[str], model: Any) -> Generator[ndarray, None, None]:
        # Tokenize the batch
        encoded = self._tokenizer.batch_encode_plus(
            batch,
            add_special_tokens=True,
            padding="longest",
        )

        # Get input IDs and attention mask
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Run inference with ONNX model
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # Get embeddings (assuming the model outputs a tuple where the first element is the embeddings)
        embeddings = model.run(None, ort_inputs)[0]

        # Process each sequence in the batch
        for seq_num in range(len(embeddings)):
            input_id = input_ids[seq_num]

            # Remove special tokens
            if hasattr(self._tokenizer, "get_special_tokens_mask"):
                special_tokens_mask = self._tokenizer.get_special_tokens_mask(
                    input_id,
                    already_has_special_tokens=True
                )
                indices_to_remove = [
                    index for index, mask in enumerate(special_tokens_mask)
                    if mask != 0
                ]
                embedding = np.delete(embeddings[seq_num], indices_to_remove, axis=0)
            else:
                embedding = embeddings[seq_num]

            yield embedding

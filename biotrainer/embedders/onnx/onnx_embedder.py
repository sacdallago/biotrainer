import onnx
import torch
import numpy as np

from pathlib import Path
from numpy import ndarray
from transformers import PreTrainedTokenizer
from typing import Union, List, Any, Generator

from ..interfaces import CustomTokenizer, EmbedderWithFallback

from ...utilities import is_device_cuda, is_device_mps, get_device_memory, get_logger

logger = get_logger(__name__)


class OrtSessionWrapper:
    def __init__(self, session):
        self.session = session
        self._is_cpu_provider = self._check_if_cpu_provider()

    def to(self, device):
        return self

    def _check_if_cpu_provider(self):
        """Check if the primary execution provider is CPU"""
        # Get the active provider
        # (first provider in the list is the primary one being used)
        providers = self.session.get_providers()
        return providers[0] == 'CPUExecutionProvider'

    def run(self, output_names, input_feed, run_options=None):
        # For CPU sessions, use strategic processing
        if self._is_cpu_provider:
            return self._run_with_memory_safety(output_names, input_feed, run_options)
        return self.session.run(output_names, input_feed=input_feed, run_options=run_options)

    def _run_with_memory_safety(self, output_names, input_feed, run_options=None):
        """Run inference with CPU memory safety considerations"""
        input_size = self._estimate_input_size(input_feed)  # GB
        safety_factor = 0.6  # Conservative safety factor considering large model architectures
        max_memory = get_device_memory(device=torch.device('cpu')) * safety_factor

        if input_size > max_memory:
            raise RuntimeError(f"Failed to use ONNX CPU Runtime to embed input of size {input_size}, "
                               f"because it exceeds the memory limit of {max_memory} GB.")

        return self.session.run(output_names, input_feed=input_feed, run_options=run_options)

    @staticmethod
    def _estimate_input_size(input_feed) -> float:
        """Roughly estimate input size in bytes"""
        minimum_size = 4.0   # At least 4GB for any inference
        gb_factor = (1024 ** 3)
        mb_factor = (1024 ** 2)

        # 1. Base input size
        sequence_length = 1000  # Reasonable default value
        for key, tensor in input_feed.items():
            if isinstance(tensor, list):
                if key == 'input_ids':
                    sequence_length += len(tensor[0])

        input_size = sequence_length * mb_factor  # Estimating 1MB/residue

        # 2. Model complexity factor - Transformer models use O(n²) memory for attention
        complexity_factor = 1.5

        # 3. Model base overhead (the ONNX model itself plus runtime overhead)
        model_base_overhead_gb = 2.0  # Assume usually about 2GB for model weights and runtime

        # Calculate total memory estimate
        estimated_memory_gb = (
                (input_size / gb_factor) * complexity_factor
                + model_base_overhead_gb
        )

        return max(estimated_memory_gb, minimum_size)


class OnnxEmbedder(EmbedderWithFallback):

    def __init__(self, onnx_path: Path, tokenizer: PreTrainedTokenizer, device: Union[str, torch.device]):
        try:
            import onnxruntime as ort
            self._onnxruntime = ort
            self._check_onnxruntime_gpu()
        except ImportError:
            raise Exception("No onnxruntime in current environment found! Please install one first, e.g. "
                            "uv pip install -e '.[onnx-cpu]'!")

        self.name = f"onnx-{onnx_path.name.replace('.onnx', '')}"
        self._onnx_path = onnx_path
        self._device = device
        self._model = self._load_model()

        self._tokenizer = tokenizer
        if isinstance(tokenizer, CustomTokenizer):
            self._preprocessing_strategy = tokenizer.preprocessing_strategy

    def _check_onnxruntime_gpu(self):
        if torch.cuda.is_available():
            if 'CUDAExecutionProvider' not in self._onnxruntime.get_available_providers():
                logger.info("CUDA is available but onnxruntime-gpu is not installed. "
                      "You can install it with: uv pip install -e '.[onnx-gpu]'")
        if torch.mps.is_available():
            if 'CoreMLExecutionProvider' not in self._onnxruntime.get_available_providers():
                logger.info("MPS is available but onnxruntime-coreml is not installed. "
                      "You can install it with: uv pip install -e '.[onnx-mac]'")

    def _load_model(self) -> OrtSessionWrapper:
        # Create ONNX Runtime session
        onnx_model = onnx.load(self._onnx_path)
        onnx.checker.check_model(onnx_model)

        if is_device_cuda(self._device):
            ep_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif is_device_mps(self._device):
            ep_list = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            ep_list = ['CPUExecutionProvider']

        ort_session = self._onnxruntime.InferenceSession(self._onnx_path, providers=ep_list)

        return OrtSessionWrapper(ort_session)

    def _get_fallback_model(self) -> OrtSessionWrapper:
        onnx_model = onnx.load(self._onnx_path)
        onnx.checker.check_model(onnx_model)

        ep_list = ['CPUExecutionProvider']

        ort_session = self._onnxruntime.InferenceSession(self._onnx_path, providers=ep_list)
        return OrtSessionWrapper(ort_session)

    def _embed_single(self, sequence: str) -> ndarray:
        [embedding] = self._embed_batch([sequence])
        return embedding

    def _embed_batch_implementation(self, batch: List[str], model: Any) -> Generator[torch.tensor, None, None]:
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

        # Get embeddings
        try:
            embeddings = model.run(None, ort_inputs)[0]
        except Exception as e:
            raise RuntimeError(e)

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

            yield torch.tensor(embedding)

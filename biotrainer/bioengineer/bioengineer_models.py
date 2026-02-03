import torch

from typing import Iterable, Optional, Callable, List, Tuple
from transformers import EsmForMaskedLM, EsmTokenizer, BertTokenizer, BertForMaskedLM, AutoTokenizer, \
    AutoModelForCausalLM

from .bioengineer_interfaces import BertLikeEngineer, GPTLikeEngineer

from ..embedders.huggingface.ESM2 import _esm2_family_dict
from ..embedders.interfaces import preprocess_sequences_with_whitespaces, preprocess_sequences_without_whitespaces


class ESM2Engineer(BertLikeEngineer):
    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        if embedder_name in _esm2_family_dict.keys():
            tokenizer = EsmTokenizer.from_pretrained(embedder_name, do_lower_case=False)
            model = EsmForMaskedLM.from_pretrained(embedder_name).to(device).eval()
            return cls(name=embedder_name, model=model, tokenizer=tokenizer)
        return None

    def _find_preprocessing_strategy(self) -> Callable:
        return preprocess_sequences_without_whitespaces


class ProtBertEngineer(BertLikeEngineer):
    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        if "prot_bert" in embedder_name:
            tokenizer = BertTokenizer.from_pretrained(embedder_name, do_lower_case=False)
            model = BertForMaskedLM.from_pretrained(embedder_name).to(device).eval()
            return cls(name=embedder_name, model=model, tokenizer=tokenizer)
        return None

    def _find_preprocessing_strategy(self) -> Callable:
        return preprocess_sequences_with_whitespaces


class ProtGPT2Engineer(GPTLikeEngineer):
    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        if embedder_name == "nferruz/ProtGPT2":
            tokenizer = AutoTokenizer.from_pretrained(embedder_name)
            model = AutoModelForCausalLM.from_pretrained(embedder_name).to(device).eval()
            return cls(name=embedder_name, model=model, tokenizer=tokenizer)
        return None

    def _find_preprocessing_strategy(self) -> Callable:
        def protgpt_preprocessing(sequences: Iterable[str], mask_token: Optional[str]):
            result = []
            for seq in sequences:
                # Split into chunks of 60 characters, then add newline characters between them
                seq_60 = "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60))
                formatted = f"<|endoftext|>\n{seq_60}\n<|endoftext|>"
                result.append(formatted)
            return result
        return protgpt_preprocessing

    def _tokenize(self, batch: List[str], preprocess: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if preprocess:
            batch = self._preprocess_sequences(batch)

        ids = self._tokenizer.batch_encode_plus(
            batch,
            is_split_into_words=False,
        )  # No padding, no special tokens

        tokenized_sequences = torch.tensor(ids["input_ids"]).to(self._model.device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(self._model.device)
        return tokenized_sequences, attention_mask

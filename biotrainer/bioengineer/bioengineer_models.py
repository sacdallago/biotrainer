import torch

from transformers import EsmForMaskedLM, EsmTokenizer, BertTokenizer, BertForMaskedLM

from .bioengineer_interfaces import BertLikeEngineer

from ..embedders.huggingface.ESM2 import _esm2_family_dict
from ..embedders.interfaces import preprocess_sequences_with_whitespaces, preprocess_sequences_without_whitespaces


class ESM2Engineer(BertLikeEngineer):
    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        if embedder_name in _esm2_family_dict.keys():
            tokenizer = EsmTokenizer.from_pretrained(embedder_name, do_lower_case=False)
            model = EsmForMaskedLM.from_pretrained(embedder_name).to(device)
            model = model.eval()
            return cls(name=embedder_name, model=model, tokenizer=tokenizer)
        return None

    def _find_preprocessing_strategy(self):
        return preprocess_sequences_without_whitespaces


class ProtBertEngineer(BertLikeEngineer):
    @classmethod
    def detect(cls, embedder_name: str, device: torch.device):
        if "prot_bert" in embedder_name:
            tokenizer = BertTokenizer.from_pretrained(embedder_name, do_lower_case=False)
            model = BertForMaskedLM.from_pretrained(embedder_name).to(device)
            model = model.eval()
            return cls(name=embedder_name, model=model, tokenizer=tokenizer)
        return None

    def _find_preprocessing_strategy(self):
        return preprocess_sequences_with_whitespaces

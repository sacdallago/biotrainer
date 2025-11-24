import torch
import pickle

from pathlib import Path
from typing import Dict, Optional, Any

from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FeatureScaler:

    def __init__(self, method: str, loaded_scaler: Optional[Any] = None):
        self.method = method
        self.scaler = loaded_scaler
    
    @classmethod
    def load(cls, method: str, load_path: Path):
        with open(load_path, "rb") as f:
            loaded_scaler = pickle.load(f)
        return cls(method=method, loaded_scaler=loaded_scaler)
    
    @staticmethod
    def methods():
        return {"none": None,
                "standard": StandardScaler(),
                "minmax": MinMaxScaler()}

    def fit(self, x: Dict[str, torch.tensor]):
        self.scaler = self.methods().get(self.method, None)
        assert self.scaler is not None, f"Feature scaler called with unknown method: {self.method}"

        embeddings = torch.stack(list(x.values())).numpy()
        self.scaler.fit(embeddings)
        return self

    def transform(self, x: Dict[str, torch.tensor]):
        assert self.scaler is not None, f"Feature scaler called without fitting!"
        
        embeddings = torch.stack(list(x.values())).numpy()
        scaled_embeddings = self.scaler.transform(embeddings)
        return {seq_id: torch.tensor(scaled_emb) for seq_id, scaled_emb in zip(x.keys(), scaled_embeddings)}
    
    def save(self, save_path: Path):
        with open(save_path, "wb") as f:
            pickle.dump(self.scaler, f)


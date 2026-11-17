from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any

from ..data_classes import Protocol
from .base import OpenAPISerializableBaseModel
from ..utils import constants as biotrainer_constants


class BiotrainerInferenceResult(OpenAPISerializableBaseModel):
    predictions: List[BiotrainerPrediction] = Field(description="List of predictions")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Metrics")

    def revert_mappings(self, protocol: Protocol, class_int2str: Dict[int, str]) -> BiotrainerInferenceResult:
        mapped_predictions = [pred.revert_mappings(protocol, class_int2str) for pred in self.predictions]
        return BiotrainerInferenceResult(predictions=mapped_predictions, metrics=self.metrics)

    def omit_predictions(self) -> BiotrainerInferenceResult:
        # Necessary to drop predictions for saving in file for save_split_ids=False
        return BiotrainerInferenceResult(predictions=[], metrics=self.metrics)

    def prediction_by_id(self, seq_id: str) -> Optional[BiotrainerPrediction]:
        return next((pred for pred in self.predictions if pred.seq_id == seq_id), None)

    def replace_seq_ids(self, hash2id: Dict[str, str]):
        return BiotrainerInferenceResult(
            predictions=[pred.replace_seq_id(hash2id.get(pred.seq_id)) for pred in self.predictions],
            metrics=self.metrics)


class BiotrainerPrediction(OpenAPISerializableBaseModel):
    seq_id: str = Field(description="Sequence identifier")
    prediction: Union[str, float, list] = Field(description="Predicted value")
    is_aggregated: bool = Field(default=False,
                                description="Whether the prediction is an aggregated per-residue prediction")
    residue_index: Optional[int] = Field(default=None,
                                         description="Residue index for non-collapsed per-residue predictions")
    raw_prediction: Optional[Union[str, float, list]] = Field(default=None, description="Raw prediction of the model")
    mcd_predictions: Optional[List[Any]] = Field(default=None, description="All Monte-Carlo-Dropout predictions")
    mcd_mean: Optional[Union[float, List[float]]] = Field(default=None, description="Monte-Carlo-Dropout mean(s)")
    mcd_std: Optional[Union[float, List[float]]] = Field(default=None,
                                                         description="Monte-Carlo-Dropout standard deviation(s)")
    mcd_lower_bound: Optional[Union[float, List[float]]] = Field(default=None,
                                                                 description="Monte-Carlo-Dropout lower bound(s)")
    mcd_upper_bound: Optional[Union[float, List[float]]] = Field(default=None,
                                                                 description="Monte-Carlo-Dropout upper bound(s)")
    bald_score: Optional[float] = Field(default=None, description="BALD score")

    def revert_mappings(self, protocol: Protocol,
                        class_int2str: Optional[Dict[int, str]] = None) -> BiotrainerPrediction:
        if class_int2str is None:
            return self

        pred = self.prediction
        mcd_preds = self.mcd_predictions
        raw_pred = self.raw_prediction
        is_aggregated = False
        if protocol in Protocol.classification_protocols():
            if isinstance(pred, list):  # per-residue
                delimiter = ""  # classification delimiter
                pred = delimiter.join([class_int2str[int(pred_idx)] for pred_idx in pred])
                mcd_preds = [delimiter.join([class_int2str[int(mcd_pred_idx)] for mcd_pred_idx in mcd_pred])
                             for mcd_pred in mcd_preds] if mcd_preds is not None else None
                is_aggregated = True
            else:
                pred = class_int2str[int(pred)]
                mcd_preds = [class_int2str[int(mcd_pred)] for mcd_pred in mcd_preds] if mcd_preds is not None else None

        return BiotrainerPrediction(seq_id=self.seq_id,
                                    prediction=pred,
                                    raw_prediction=raw_pred,
                                    is_aggregated=is_aggregated,
                                    mcd_predictions=mcd_preds,
                                    mcd_mean=self.mcd_mean, mcd_std=self.mcd_std,
                                    mcd_lower_bound=self.mcd_lower_bound,
                                    mcd_upper_bound=self.mcd_upper_bound,
                                    bald_score=self.bald_score
                                    )

    @staticmethod
    def aggregate_per_residue_predictions(predictions: List[BiotrainerPrediction],
                                          protocol: Protocol) -> List[BiotrainerPrediction]:
        """ Aggregate predictions into a single prediction for each sequence"""

        if any(pred.is_aggregated for pred in predictions):
            raise ValueError("Cannot aggregate predictions that are already aggregated!")

        if any(pred.residue_index is None for pred in predictions):
            raise ValueError("Cannot aggregate predictions without residue indices!")

        seq_preds = {}
        for pred in predictions:
            if pred.seq_id not in seq_preds:
                seq_preds[pred.seq_id] = []
            seq_preds[pred.seq_id].append(pred)

        delimiter = biotrainer_constants.RESIDUE_TO_VALUE_TARGET_DELIMITER if protocol in Protocol.regression_protocols() else ""
        return [BiotrainerPrediction(seq_id=seq_id,
                                     is_aggregated=True,
                                     prediction=delimiter.join([str(pred.prediction) for pred in
                                                                sorted(preds, key=lambda p: p.residue_index)]))
                for seq_id, preds in seq_preds.items()]

    def replace_seq_id(self, new_seq_id: Optional[str] = None) -> BiotrainerPrediction:
        """ Useful to undo sequence hash mapping to original sequence id """
        return BiotrainerPrediction(seq_id=new_seq_id or self.seq_id,
                                    prediction=self.prediction,
                                    is_aggregated=self.is_aggregated,
                                    residue_index=self.residue_index,
                                    raw_prediction=self.raw_prediction,
                                    mcd_predictions=self.mcd_predictions,
                                    mcd_mean=self.mcd_mean, mcd_std=self.mcd_std,
                                    mcd_lower_bound=self.mcd_lower_bound,
                                    mcd_upper_bound=self.mcd_upper_bound,
                                    bald_score=self.bald_score)

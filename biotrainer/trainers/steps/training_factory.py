import torch
import random

from typing import List, Dict, Any
from torch.utils.data import DataLoader

from ..pipeline import PipelineContext

from ...losses import get_loss
from ...utilities import get_logger
from ...optimizers import get_optimizer
from ...solvers import get_solver, Solver
from ...models import get_model, FineTuningModel
from ...datasets import get_dataset, get_embeddings_collate_function, BiotrainerDataset

logger = get_logger(__name__)


class TrainingFactory:
    @staticmethod
    def create_dataset(context: PipelineContext, split: List, mode: str, finetuning: bool = False) -> BiotrainerDataset:
        # Apply limited sample number
        limited_sample_size = context.config["limited_sample_size"]
        if mode == "train" and limited_sample_size and limited_sample_size > 0:
            logger.info(f"Using limited sample size of {limited_sample_size} for training dataset")
            split = random.sample(split,
                                  k=min(limited_sample_size, len(split)))

        random_masking = context.config.get("finetuning_config", {}).get("random_masking", False)
        mask_token = context.embedding_service._embedder.get_mask_token() if context.embedding_service is not None else None
        return get_dataset(samples=split,
                           finetuning=finetuning,
                           random_masking=random_masking,
                           mask_token=mask_token,
                           class_str2int=context.class_str2int
                           )

    @staticmethod
    def create_dataloader(context: PipelineContext, dataset, hyper_params: Dict,
                          finetuning: bool = False) -> DataLoader:
        # Create dataloader from dataset
        if finetuning:
            collate_fn = lambda batch: (
                [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch], [len(x[1]) for x in batch])
        else:
            collate_fn = get_embeddings_collate_function(context.config["protocol"])
        return DataLoader(
            dataset=dataset, batch_size=hyper_params["batch_size"], shuffle=hyper_params["shuffle"], drop_last=False,
            collate_fn=collate_fn
        )

    @staticmethod
    def create_solver(context: PipelineContext, split_name: str, model, loss_function, optimizer,
                      hyper_params: Dict) -> Solver:
        return get_solver(protocol=context.config["protocol"], name=split_name, network=model, optimizer=optimizer,
                          loss_function=loss_function,
                          output_manager=context.output_manager,
                          device=context.config["device"],
                          number_of_epochs=hyper_params["num_epochs"],
                          patience=hyper_params["patience"], epsilon=hyper_params["epsilon"],
                          log_dir=hyper_params["log_dir"], n_classes=context.n_classes)

    @staticmethod
    def create_model_loss_optimizer(context: PipelineContext,
                                    hyper_params: Dict[str, Any]) -> (
            torch.nn.Module, torch.nn.Module, torch.nn.Module):
        # Initialize model
        model = get_model(n_classes=context.n_classes, n_features=context.n_features,
                          **hyper_params)

        if "finetuning_config" in context.config:
            model = FineTuningModel(embedding_service=context.embedding_service,
                                    downstream_model=model,
                                    collate_fn=get_embeddings_collate_function(context.config["protocol"]),
                                    protocol=context.config["protocol"],
                                    device=context.config["device"], )

        # Initialize loss function
        loss_function = get_loss(weight=context.class_weights, **hyper_params)

        # Initialize optimizer
        optimizer = get_optimizer(model_parameters=model.parameters(), **hyper_params)

        return model, loss_function, optimizer

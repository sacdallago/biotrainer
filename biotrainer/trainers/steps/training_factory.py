import torch
import random

from typing import List, Dict, Any
from torch.utils.data import DataLoader

from ..pipeline import PipelineContext

from ...losses import get_loss
from ...models import get_model
from ...optimizers import get_optimizer
from ...solvers import get_solver, Solver
from ...utilities import DatasetSample, get_logger
from ...datasets import get_dataset, get_collate_function

logger = get_logger(__name__)


class TrainingFactory:
    @staticmethod
    def create_embeddings_dataset(context: PipelineContext, split: List[DatasetSample], mode: str):
        # Apply limited sample number
        limited_sample_size = context.config["limited_sample_size"]
        if mode == "train" and limited_sample_size and limited_sample_size > 0:
            logger.info(f"Using limited sample size of {limited_sample_size} for training dataset")
            split = random.sample(split,
                                  k=min(limited_sample_size, len(split)))
        return get_dataset(context.config["protocol"], split)

    @staticmethod
    def create_dataloader(context: PipelineContext, dataset, hyper_params: Dict) -> DataLoader:
        # Create dataloader from dataset
        return DataLoader(
            dataset=dataset, batch_size=hyper_params["batch_size"], shuffle=hyper_params["shuffle"], drop_last=False,
            collate_fn=get_collate_function(context.config["protocol"])
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

        # Initialize loss function
        loss_function = get_loss(weight=context.class_weights, **hyper_params)

        # Initialize optimizer
        optimizer = get_optimizer(model_parameters=model.parameters(), **hyper_params)

        return model, loss_function, optimizer

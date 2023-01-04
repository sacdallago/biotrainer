import torch
import numpy as np

from copy import deepcopy
from torch.utils.data import DataLoader
from typing import Union, Optional, Dict, Iterable, Tuple, Any

from ..losses import get_loss
from ..models import get_model
from ..solvers import get_solver
from ..optimizers import get_optimizer
from ..trainers import revert_mappings
from ..utilities import get_device, DatasetSample
from ..datasets import get_dataset, get_collate_function


class Inferencer:

    def __init__(
            self,
            # Constant parameters for all split solvers
            protocol: str,
            embedder_name: str,
            # Optional constant parameters
            class_int_to_string: Optional[Dict[int, str]] = None,
            device: Union[None, str, torch.device] = None,
            # Everything else
            **kwargs
    ):
        self.protocol = protocol
        self.device = get_device(device)
        self.embedder_name = embedder_name
        self.class_int2str = class_int_to_string
        self.collate_function = get_collate_function(protocol)

        self.solvers_and_loaders_by_split = self._create_solvers_and_loaders_by_split(**kwargs)
        print(f"Got {len(self.solvers_and_loaders_by_split.keys())} split(s): "
              f"{', '.join(self.solvers_and_loaders_by_split.keys())}")

    def _create_solvers_and_loaders_by_split(self, **kwargs) -> Dict[str, Tuple[Any, Any]]:
        result_dict = {}
        splits = kwargs["split_results"].keys()
        for split in splits:
            # Ignore average result
            if "average" in split:
                continue
            split_config = deepcopy(kwargs)
            for key, value in kwargs["split_results"][split]["split_hyper_params"].items():
                split_config[key] = value

            # Positional arguments
            model_choice = split_config.pop("model_choice")
            n_classes = split_config.pop("n_classes")
            n_features = split_config.pop("n_features")
            loss_choice = split_config.pop("loss_choice")
            optimizer_choice = split_config.pop("optimizer_choice")
            learning_rate = split_config.pop("learning_rate")
            experiment_dir = split_config.pop("log_dir")

            model = get_model(protocol=self.protocol, model_choice=model_choice,
                              n_classes=n_classes, n_features=n_features,
                              **split_config
                              )
            loss_function = get_loss(protocol=self.protocol, loss_choice=loss_choice,
                                     device=self.device,
                                     **split_config
                                     )
            optimizer = get_optimizer(protocol=self.protocol, optimizer_choice=optimizer_choice,
                                      model_parameters=model.parameters(), learning_rate=learning_rate,
                                      **split_config
                                      )

            solver = get_solver(
                protocol=self.protocol, name=split,
                network=model, optimizer=optimizer, loss_function=loss_function, device=self.device,
                experiment_dir=experiment_dir, num_classes=n_classes
            )
            solver.load_checkpoint(resume_training=False)

            def dataloader_function(dataset):
                return DataLoader(dataset=dataset, batch_size=split_config["batch_size"],
                                  shuffle=False, drop_last=False,
                                  collate_fn=self.collate_function)

            result_dict[split] = (solver, dataloader_function)
        return result_dict

    def _load_solver_and_dataloader(self, embeddings: Iterable, split_name):
        if split_name not in self.solvers_and_loaders_by_split.keys():
            raise Exception(f"Unknown split_name {split_name} for given configuration!")

        solver, loader = self.solvers_and_loaders_by_split[split_name]
        dataset = get_dataset(self.protocol, samples=[
            DatasetSample(idx, torch.tensor(np.array(embedding)), torch.empty(1))
            for idx, embedding in enumerate(embeddings)
        ])
        dataloader = loader(dataset)
        return solver, dataloader

    def from_embeddings(self, embeddings: Iterable, split_name: str = "hold_out") -> Dict[str, Union[str, int, float]]:
        solver, dataloader = self._load_solver_and_dataloader(embeddings, split_name)

        predictions = solver.inference(dataloader)["mapped_predictions"]

        # For class predictions, revert from int (model output) to str (class name)
        predictions = revert_mappings(protocol=self.protocol, test_predictions=predictions,
                                      class_int2str=self.class_int2str)

        return predictions

    def from_embeddings_with_monte_carlo_dropout(self, embeddings: Iterable,
                                                 split_name: str = "hold_out",
                                                 n_forward_passes: int = 30,
                                                 confidence_level: float = 0.05):
        if "_value" not in self.protocol and "_interaction" not in self.protocol:
            raise Exception(f"Monte carlo dropout only implemented for x_to_value "
                            f"and protein_protein_interaction protocols!")

        solver, dataloader = self._load_solver_and_dataloader(embeddings, split_name)

        predictions = solver.inference_monte_carlo_dropout(dataloader=dataloader,
                                                           n_forward_passes=n_forward_passes,
                                                           confidence_level=confidence_level)["mapped_predictions"]

        # For class predictions, revert from int (model output) to str (class name)
        predictions = revert_mappings(protocol=self.protocol, test_predictions=predictions,
                                      class_int2str=self.class_int2str)

        return predictions

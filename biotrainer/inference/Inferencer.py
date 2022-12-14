import torch

from typing import Union, Optional, Dict, Iterable, List

from torch.utils.data import DataLoader

from ..losses import get_loss
from ..models import get_model
from ..solvers import get_solver
from ..utilities import get_device
from ..optimizers import get_optimizer
from ..trainers import revert_mappings
from ..datasets import get_dataset, get_collate_function


class Inferencer:

    def __init__(
            self,

            # These are from the original input config
            protocol: str,
            n_classes: int,
            n_features: int,
            model_choice: str,
            embedder_name: str,

            # These are from the output (out.yml) file
            log_dir: str,
            class_int_to_string: Optional[Dict[int, str]] = None,

            # Fillers
            learning_rate: float = 1e-3,
            loss_choice: str = "cross_entropy_loss",
            optimizer_choice: str = "adam", batch_size: int = 128,
            device: Union[None, str, torch.device] = None,

            # Everything else
            **kwargs
    ):
        self.protocol = protocol
        self.device = get_device(device)
        self.batch_size = batch_size
        self.class_int2str = class_int_to_string
        self.embedder_name = embedder_name

        model = get_model(
            protocol=protocol, model_choice=model_choice,
            n_classes=n_classes, n_features=n_features
        )
        loss_function = get_loss(
            protocol=protocol, loss_choice=loss_choice, device=device
        )
        optimizer = get_optimizer(
            protocol=protocol, optimizer_choice=optimizer_choice,
            learning_rate=learning_rate, model_parameters=model.parameters()
        )

        self.solver = get_solver(
            protocol, network=model, optimizer=optimizer, loss_function=loss_function, device=self.device,
            experiment_dir=log_dir, num_classes=n_classes
        )
        self.collate_function = get_collate_function(protocol)
        self.solver.load_checkpoint()

    def from_embeddings(self, embeddings: Iterable) -> Dict[str, Union[str, int, float]]:
        dataset = get_dataset(self.protocol, samples={
            idx: (torch.tensor(embedding), torch.empty(1))
            for idx, embedding in enumerate(embeddings)
        })

        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
            collate_fn=self.collate_function
        )

        predictions = self.solver.inference(dataloader)["mapped_predictions"]

        # For class predictions, revert from int (model output) to str (class name)
        predictions = revert_mappings(protocol=self.protocol, test_predictions=predictions,
                                      class_int2str=self.class_int2str)

        return predictions

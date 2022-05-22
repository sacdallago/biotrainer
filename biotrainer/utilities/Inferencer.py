import torch

from typing import Union, Optional, Dict, Iterable, List

from torch.utils.data import DataLoader

from ..optimizers import get_optimizer
from ..losses import get_loss
from ..models import get_model
from .cuda_device import get_device

# TODO: change to get_solver once Sebas' stuff is merged!
from ..solvers import ResidueSolver
from ..datasets import ResidueEmbeddingsDataset, pad_sequences


class Inferencer:

    def __init__(
            self,

            # These are from the original input config
            protocol: str,
            n_classes: int,
            n_features: int,
            model_choice: str,
            embedder_name: str,

            # These are from post-training
            log_dir: str,
            class_int_to_string: Optional[Dict[int, str]],

            # Fillers
            learning_rate: float = 1e-3,
            loss_choice: str = "cross_entropy_loss",
            optimizer_choice: str = "adam", batch_size: int = 128,
            device: Union[None, str, torch.device] = None, embeddings_file_path: str = None,
    ):

        model = get_model(
            protocol=protocol, model_choice=model_choice,
            n_classes=n_classes, n_features=n_features
        )
        loss_function = get_loss(
            protocol=protocol, loss_choice=loss_choice
        )
        optimizer = get_optimizer(
            protocol=protocol, optimizer_choice=optimizer_choice,
            learning_rate=learning_rate, model_parameters=model.parameters()
        )

        self.device = get_device(device)
        self.batch_size = batch_size
        self.dataset = ResidueEmbeddingsDataset
        self.class_int_to_string = class_int_to_string
        self.protocol = protocol
        self.embedder_name = embedder_name

        # TODO: change to get_solver once Sebas' stuff is merged!
        self.solver = ResidueSolver(
            network=model, optimizer=optimizer, loss_function=loss_function, device=self.device,
            experiment_dir=log_dir
        )
        self.solver.load_checkpoint()

    def from_embeddings(self, embeddings: Iterable) -> List[Union[str, int]]:
        dataset = self.dataset({
            idx: (torch.tensor(embedding), torch.tensor([0])) for idx, embedding in enumerate(embeddings)
        })

        if self.protocol == 'residue_to_class':
            dataloader = DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=pad_sequences
            )

            results = self.solver.inference(dataloader)

            results['predictions'] = ["".join(
                [self.class_int_to_string[p] for p in prediction]
            ) for prediction in results['predictions']]

            return results['predictions']
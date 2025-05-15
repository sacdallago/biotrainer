from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from .biotrainer_output_observer import BiotrainerOutputObserver, OutputData

class TensorboardWriter(BiotrainerOutputObserver):

    def __init__(self, log_dir: Path):
        super().__init__()
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def update(self, data: OutputData) -> None:
        if data.config:
            self.writer.add_hparams({
                'model': data.config["model_choice"],
                'num_epochs': data.config["num_epochs"],
                'use_class_weights': data.config["use_class_weights"],
                'learning_rate': data.config["learning_rate"],
                'batch_size': data.config["batch_size"],
                'embedder_name': data.config["embedder_name"],
                'seed': data.config["seed"],
                'loss': data.config["loss_choice"],
                'optimizer': data.config["optimizer_choice"],
            }, {})
        if data.training_iteration:
            split = data.training_iteration[0]  # TODO Add split to tensorboard
            epoch_metrics = data.training_iteration[1]
            self.writer.add_scalars("Epoch/train", epoch_metrics.training, epoch_metrics.epoch)
            self.writer.add_scalars("Epoch/validation", epoch_metrics.validation, epoch_metrics.epoch)
            self.writer.add_scalars("Epoch/comparison", {
                'training_loss': epoch_metrics.training['loss'],
                'validation_loss': epoch_metrics.validation['loss'],
            }, epoch_metrics.epoch)

    def close(self) -> None:
        self.writer.close()

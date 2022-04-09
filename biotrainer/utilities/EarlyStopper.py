import torch
import logging

import numpy as np
import torch.nn as nn

from pathlib import Path

logger = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, log_dir: str, patience: int = 10, epsilon: float = 0.001, min_loss: float = np.inf,
                 n_worse: int = 0):
        """

        :param log_dir: where to store the checkpoint file
        :param patience: number of max. epochs accepted for non-improving loss
        :param epsilon: the minimal difference in improvement a model needs to reach
        :param min_loss: counter for lowest/best overall-loss
        :param n_worse: counter of consecutive non-improving losses
        """
        self.checkpoint_path = Path(log_dir) / 'checkpoint.pt'
        self.epsilon = epsilon
        self.min_loss = min_loss
        self.n_worse = n_worse
        self.patience = patience

    def load_checkpoint(self, model):
        state = torch.load(self.checkpoint_path)
        model.load_state_dict(state['state_dict'])
        logger.info(f"Loaded model from epoch: {state['epoch']}")
        return model, state['epoch']

    def save_checkpoint(self, model: nn.Module, epoch: int, optimizer):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, self.checkpoint_path)

    def check_performance(self, model: nn.Module, test_loader, crit, optimizer, epoch: int, num_epochs) -> bool:
        current_loss = testing(model, test_loader, crit, epoch, num_epochs)

        # if the model improved compared to previously best checkpoint
        if current_loss < (self.min_loss - self.epsilon):
            logger.info(f'New best model found with loss={current_loss:.3f}')
            self.save_checkpoint(model, epoch, optimizer)
            self.min_loss = current_loss  # save new loss as best checkpoint
            self.n_worse = 0
        else:  # if the model did not improve further increase counter
            self.n_worse += 1
            if self.n_worse > self.patience:  # if the model did not improve for 'patience' epochs
                print('Stopping due to early stopping after epoch {}!'.format(epoch))
                return True
        return False

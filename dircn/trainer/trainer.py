from typing import Callable, Dict, Union
from collections import defaultdict
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

# from torchvision.utils import make_grid
# from base import BaseTrainer
# from utils import inf_loop, MetricTracker

from ..base import BaseTrainer
from ..config import ConfigReader
from ..models import MultiMetric, MultiLoss


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: Union[MultiLoss, Callable],
                 metric_ftns: Union[MultiMetric, Dict[str, Callable]],
                 config: ConfigReader,
                 data_loader: torch.utils.data.dataloader,
                 valid_data_loader: torch.utils.data.dataloader = None,
                 seed: int = None,
                 device: str = None,
                 log_step: int = None,
                 ):

        super().__init__(model=model,
                         loss_function=loss_function,
                         metric_ftns=metric_ftns,
                         config=config,
                         seed=seed,
                         device=device)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.images_pr_iteration = int(config['trainer']['images_pr_iteration'])
        self.val_images_pr_iteration = int(config['trainer']['val_images_pr_iteration'])

        self.len_epoch = len(data_loader) if not self.iterative else self.images_pr_iteration
        self.batch_size = data_loader.batch_size
        self.log_step = int(self.len_epoch/(4*self.batch_size)) if not isinstance(log_step, int) else log_step

    def calculate_mask(self, masked_kspace: torch.Tensor):
        x = masked_kspace[:, :, :, :, 0]**2 + masked_kspace[:, :, :, :, 1]**2
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sum(x, dim=2, keepdim=True).unsqueeze(-1)
        x = x != 0
        return x

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        losses = defaultdict(list)

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            if len(data) > 4:
                mask = self.calculate_mask(masked_kspace=data)
            else:
                mask = None

            self.optimizer.zero_grad()  # Clear before inputting to model

            if mask is not None:
                output = self.model(data, mask=mask)
            else:
                output = self.model(data)

            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()

            loss = loss.item()  # Detach loss from comp graph and moves it to the cpu
            losses['loss'].append(loss)

            if batch_idx % self.log_step == 0:
                self.logger.info('Train {}: {} {} Loss: {:.6f}'.format(
                    'Epoch' if not self.iterative else 'Iteration',
                    epoch,
                    self._progress(batch_idx),
                    loss))

            if batch_idx*self.batch_size >= self.images_pr_iteration and self.iterative:
                break

        losses['loss_func'] = str(self.loss_function)
        losses['loss'] = [np.mean(losses['loss'])]

        return losses

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.valid_data_loader is None:
            return None

        self.model.eval()
        metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                metrics['loss'].append(self.loss_function(output, target).item())

                for key, metric in self.metric_ftns.items():
                    if self.metrics_is_dict:
                        metrics[key].append(metric(output.cpu(), target.cpu()).item())
                    else:
                        metrics[key].append(metric(output, target).item())

                if batch_idx*self.batch_size >= self.val_images_pr_iteration and self.iterative:
                    break

        return metrics

    def _train_iteration(self, iteration):
        """
        Training logic after an iteration, for large datasets

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        return self._train_epoch(epoch=iteration)

    def _valid_iteration(self, iteration):
        """
        Validation logic after an iteration, for large datasets

        :param epoch: Current iteration number
        """
        return self._valid_epoch(epoch=iteration)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        elif hasattr(self.data_loader, 'batch_size'):
            current = batch_idx * self.data_loader.batch_size
            total = self.len_epoch
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

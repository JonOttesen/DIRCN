import json
from pathlib import Path
from typing import Union, Dict

import torch
import numpy as np

from ..logger import get_logger

class MetricTracker(object):
    """
    A simple class ment for storing and saving training loss history
    and validation metric history in json file
    """

    TRAINING_KEY = 'training'
    VALIDATION_KEY = 'validation'
    CONFIG_KEY = 'config'

    def __init__(self,
                 config: dict):
        """
        Args:
            config (dict): The config dict which initiates the network
        """

        self.logger = get_logger(name=__name__)

        self.results = dict()
        self.results[self.TRAINING_KEY] = dict()
        self.results[self.VALIDATION_KEY] = dict()
        self.results[self.CONFIG_KEY] = config
        self.iterative = bool(config['trainer']['iterative'])

    def __getitem__(self, key):
        return self.results[key]

    def training(self):
        return self[self.TRAINING_KEY]

    def validation(self):
        return self[self.VALIDATION_KEY]

    def config(self):
        return self[self.CONFIG_KEY]

    def resume(self, resume_path: Union[str, Path]):
        """
        Resumes MetricTracker from previous state
        NB! Overwrites anything stored except the config dict
        Args:
            resume_path (str, pathlib.Path): The previous saved MetricTracker object
        """
        if not isinstance(resume_path, (str, Path)):
            TypeError('resume_path is not of type str or Path but {}'.format(type(resume_path)))

        if not Path(resume_path).is_file():
            self.logger.warning('{} is not a file will not resume '
                                'from MetricTracker instance.'.format(str(resume_path)))
        with open(str(resume_path), 'r') as inifile:
            prev = json.load(inifile)

        if self.TRAINING_KEY not in prev.keys() or self.VALIDATION_KEY not in prev.keys():
            self.logger.warning('The given file does not have the training or validation key, '
                                'will not resume from prior checkpoint.')
            return

        if self.CONFIG_KEY in prev.keys():
            if prev[self.CONFIG_KEY] != self.results[self.CONFIG_KEY]:
                self.logger.warning('Non identical configs found, '
                                    'this instance will store the new config.')

        self.results[self.TRAINING_KEY].update(prev[self.TRAINING_KEY])
        self.results[self.VALIDATION_KEY].update(prev[self.VALIDATION_KEY])

    def training_update(self,
                        loss: Dict[str, list],
                        epoch: int):
        """
        Appends new training history
        Args:
            loss (list, np.ndarray, torch.Tensor): The loss history for this batch
            epoch (int): The epoch or iteration number, repeated numbers will overwrite
                         previous history
        """

        epoch = '{}_{}'.format(
            'epoch' if not self.iterative else 'iteration',
            epoch)
        self.results[self.TRAINING_KEY][epoch] = loss

    def validation_update(self,
                          metrics: Dict[str, list],
                          epoch: int):
        """
        Appends new validation history
        Args:
            metrics (dict): A dict matching the metric to the score for one/multiple metrics
            epoch (int): The epoch or iteration number, repeated numbers will overwrite
                         previous history
        """

        epoch = '{}_{}'.format(
            'epoch' if not self.iterative else 'iteration',
            epoch)
        self.results[self.VALIDATION_KEY][epoch] = metrics

    def training_metric(self, epoch):
        epoch = '{}_{}'.format('epoch' if not self.iterative else 'iteration', epoch)
        return self.results[self.TRAINING_KEY][epoch]

    def validation_metric(self, epoch):
        epoch = '{}_{}'.format('epoch' if not self.iterative else 'iteration', epoch)
        return self.results[self.VALIDATION_KEY][epoch]

    def write_to_file(self, path: Union[str, Path]):
        """
        Writes MetricTracker to file
        Args:
            path (str, pathlib.Path): Path where the file is stored,
                                      remember to have .json suffix
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)  # Missing parents are quite the letdown
        path = str(path)

        with open(path, 'w') as outfile:
            json.dump(self.results, outfile, indent=4)

    @staticmethod
    def from_json(path: Union[str, Path]):
        """
        Returns the metrics without the config
        """
        with open(str(path), 'r') as inifile:
            prev = json.load(inifile)
        metrics = MetricTracker(config=prev[MetricTracker.CONFIG_KEY])
        metrics.resume(path)
        return metrics


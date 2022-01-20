import json
import random

from pathlib import Path
from typing import Union, List
from copy import deepcopy
import numpy as np
from copy import deepcopy

from tqdm import tqdm

from ..logger import get_logger

from ..dataset import DatasetContainer

class ModelEntry(object):

    def __init__(self,
                 model_name: str = None,
                 model_path: Union[Path, str] = None,
                 model_description: str = None,
                 dataset_container: DatasetContainer = None,
                 keyword: str = '',
                 ):
        self.logger = get_logger(name=__name__)
        self.model_name = model_name
        self.model_path = str(model_path)
        self.model_description = model_description
        self.dataset_container = dataset_container
        self.keyword = keyword

    def __str__(self):
        return str(self.to_dict())

    def __getitem__(self, index):
        return self.dataset_container[index]

    def __len__(self):
        return len(self.dataset_container)

    def add_dataset_container(self, dataset_container: DatasetContainer):
        assert isinstance(dataset_container, DatasetContainer), 'input must be DatasetContainer not {}'.format(type(dataset_container))
        self.dataset_container = deepcopy(dataset_container)

    def keys(self):
        return self.to_dict().keys()

    def to_dict(self) -> dict:
        """
        returns:
            dict format of this class
        """
        if isinstance(self.dataset_container, DatasetContainer):
            dataset_container = self.dataset_container.to_dict()
        else:
            dataset_container = self.dataset_container
        return {'model_name': self.model_name,
                'model_path': str(self.model_path),
                'model_description': self.model_description,
                'keyword': self.keyword,
                'dataset_container': dataset_container,
                }

    def from_dict(self, in_dict: dict):
        """
        Args:
            in_dict: dict, dict format of this class
        Fills in the variables from the dict
        """
        if isinstance(in_dict, dict):
            self.model_name = in_dict['model_name']
            self.model_path = in_dict['model_path']
            self.model_description = in_dict['model_description']
            self.keyword = in_dict['keyword']
            dataset_container = in_dict['dataset_container']
        if isinstance(dataset_container, dict):
            self.dataset_container = DatasetContainer()
            self.dataset_container.from_dict(dataset_container)

        return self


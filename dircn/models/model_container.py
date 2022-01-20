import json
import random

from pathlib import Path
from typing import Union, List
from copy import deepcopy
import numpy as np
from copy import deepcopy

from tqdm import tqdm

from ..logger import get_logger

from .model_entry import ModelEntry

class ModelContainer(object):

    def __init__(self, entries: List[ModelEntry] = None):
        self.logger = get_logger(name=__name__)
        self.entries = entries if entries is not None else list()

    def add_entry(self, entry: ModelEntry):
        self.entries.append(deepcopy(entry))

    def __getitem__(self, index):
        return self.entries[index]

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        container_dict = dict()
        container_dict['entries'] = [entry.to_dict() for entry in self.entries]
        return container_dict

    def from_dict(self, in_dict):
        for entry in in_dict['entries']:
            self.entries.append(ModelEntry().from_dict(entry))
        return self

    def to_json(self, path: Union[str, Path]):
        path = Path(path)
        suffix = path.suffix
        if suffix != '.json':
            raise NameError('The path must have suffix .json not, ', suffix)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as outfile:
            json.dump(self.to_dict(), outfile, ensure_ascii=False, indent=4)

    @classmethod
    def from_json(cls, path: Union[str, Path]):
        with open(path) as json_file:
            data = json.load(json_file)
        new_container = cls()
        new_container.from_dict(data)
        return new_container

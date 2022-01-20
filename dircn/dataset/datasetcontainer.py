import json
import random
import contextlib

from pathlib import Path
from typing import Union, List
from copy import deepcopy
import numpy as np
from copy import deepcopy

from tqdm import tqdm

from .datasetentry import DatasetEntry
from .datasetinfo import DatasetInfo
from ..logger import get_logger


@contextlib.contextmanager
def temp_seed(seed):
    """
    Source:
    https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


class DatasetContainer(object):
    """
    Method to save and store the information about the dataset
    This includes the shape, sequence type, locations, post/pre contrast etc.
    This means that all files can be stored at the same place or separately,
    it is cleaner to handle than just lumping files together in folders.

    And it is iterable and savable
    """

    def __init__(self,
                 info: List[DatasetInfo] = None,
                 entries: List[DatasetEntry] = None):
        """
        Args:
            info (list(DatasetInfo)): list of information about the dataset used
            entries (list(DatasetEntries)): list of the entries, i.e., the files of the dataset
        """

        self.logger = get_logger(name=__name__)
        self.info = info if info is not None else list()
        self.entries = entries if entries is not None else list()

    def __getitem__(self, index):
        return self.entries[index]

    def __len__(self):
        return len(self.entries)

    def __delitem__(self, index):
        del self.entries[index]

    def __str__(self):
        return str(self.to_dict())

    def split(self, seed: int, split: float = 0.5):
        """
        Randomly split the dataset
        Args:
            seed (int): The seed used for the random split
            split (float): The split fraction
        returns:
            The two split dataset containers
        """
        dataset = deepcopy(self)
        dataset.shuffle(seed=seed)

        split_1 = DatasetContainer()
        split_2 = DatasetContainer()
        for info in self.info:
            split_1.add_info(deepcopy(info))
            split_2.add_info(deepcopy(info))

        split_number = split*len(dataset)

        for i, entry in enumerate(dataset):
            i += 1
            if i <= split_number:
                split_1.add_entry(deepcopy(entry))
            else:
                split_2.add_entry(deepcopy(entry))
        return split_1, split_2

    def shuffle(self, seed=None):
        """
        Shuffles the entries, used for random training
        Args:
            seed (int): The seed used for the random shuffle
        """
        with temp_seed(seed):
            random.shuffle(self.entries)

    def info_dict(self):
        """
        returns:
            dict version of the list of DatasetInfo
        """
        info_dict = dict()
        for inf in self.info:
            info_dict[inf.datasetname] = inf.to_dict()

        return info_dict

    def copy(self):
        return deepcopy(self.entries)

    def add_info(self, info: DatasetInfo):
        """
        Append DatasetInfo
        Args:
            info (DatasetInfo): The DatasetInfo to be appended
        """
        self.info.append(deepcopy(info))

    def add_entry(self, entry: DatasetEntry):
        """
        Append DataseEntry
        Args:
            info (DatasetEntry): The DatasetEntry to be appended
        """
        self.entries.append(deepcopy(entry))

    def add_shapes(self, open_func=None, shape=None, keyword='kspace'):
        """
        Fetches the shapes of the images for each entry
        Args:
            open_func (callable): The function used to open the files
            shape (tuple): The shape of the entry, this can be image size
            keyword: Potential key used for hdf5 files
        """
        for entry in tqdm(self):
            entry.add_shape(open_func=open_func, shape=shape, keyword=keyword)

    def shapes_given(self):
        """
        Checks if all entries have shapes in the DatasetContainer
        """
        for entry in self:
            if entry.shape is None:
                return False
            else:
                return True

    def shapes(self):
        return list(set([tuple(entry['shape']) for entry in self]))

    def sequences(self):
        return list(set([entry['sequence_type'] for entry in self]))

    def keys(self):
        return self.to_dict().keys()

    def to_dict(self):
        """
        returns:
            dict version of DatasetContainer
        """
        container_dict = dict()
        container_dict['info'] = [inf.to_dict() for inf in self.info]
        container_dict['entries'] = [entry.to_dict() for entry in self.entries]
        return container_dict

    def from_dict(self, in_dict):
        """
        Appends data into DatasetContainer from dict
        Args:
            in_dict (dict): Dict to append data from, meant to be used to recover when loading from file
        """
        for inf in in_dict['info']:
            self.info.append(DatasetInfo().from_dict(inf))
        for entry in in_dict['entries']:
            self.entries.append(DatasetEntry().from_dict(entry))

    def to_json(self, path: Union[str, Path]):
        """
        Save DatasetContainer as json file
        Args:
            path (str, Path): path where DatasetContainer is saved
        """
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
        """
        Load DatasetContainer from file
        Args:
            path (str, Path): Path to load from
        returns:
            The DatasetContainer from file
        """
        with open(path) as json_file:
            data = json.load(json_file)
        new_container = cls()
        new_container.from_dict(data)
        return new_container

    def from_folder(self,
                    path: Union[str, Path],
                    datasetname: str,
                    dataset_type: str,
                    source: str = 'Some source',
                    dataset_description: str = 'Some description',
                  ):
        """
        Add multiple files from folder
        Args:
            path (str, Path): Path to folder
            datasetname (str): Name of dataset
            dataset_type (str): The type of dataset this is
            source (str): The source of the dataset (fastMRI)
            dataset_description (str): description of dataset
        returns:
            DatasetContainer from folder
        """
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError('path argument is {}, expected type is pathlib.Path or str'.format(type(path)))

        path = path.absolute()
        files = list(path.glob('*'))

        info = DatasetInfo(
            datasetname=datasetname,
            dataset_type=dataset_type,
            source=source,
            dataset_description=dataset_description
            )

        self.add_info(info=info)

        for file in tqdm(files):
            filename = file.name

            entry = DatasetEntry(
                image_path=file,
                datasetname=datasetname,
                dataset_type=dataset_type,
                )

            self.add_entry(entry=entry)

        self.add_shapes()

        return self


    def fastMRI(self,
                path: Union[str, Path],
                datasetname: str,
                dataset_type: str,
                source: str = 'fastMRI',
                dataset_description: str = 'Data for fastMRI challenge',
                multicoil=True):

        """
        Fills up the container using the folder containing the fastMRI data
        Args:
            path (str, Path): Path to fastMRI data
            datasetname (str): Name of dataset
            dataset_type (str): The type of dataset this is
            source (str): The source of the dataset (fastMRI)
            dataset_description (str): description of dataset
        returns:
            DatasetContainer filled with fastMRI data
        """
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError('path argument is {}, expected type is pathlib.Path or str'.format(type(path)))

        path = path.absolute()
        files = list(path.glob('*.h5'))

        info = DatasetInfo(
            datasetname=datasetname,
            dataset_type=dataset_type,
            source=source,
            dataset_description=dataset_description
            )

        self.add_info(info=info)

        for file in tqdm(files):
            filename = file.name

            entry = DatasetEntry(
                image_path=file,
                datasetname=datasetname,
                dataset_type=dataset_type,
                multicoil=multicoil
                )

            # Pre or post contrast or None if neither
            if 'PRE' in filename or 'pre' in filename or 'Pre' in filename:
                entry.pre_contrast = True
            elif 'POST' in filename or 'post' in filename or 'Post' in filename:
                entry.post_contrast = True

            if 'T2' in filename or 't2' in filename:
                entry.sequence_type = 'T2'
            elif 'T1' in filename or 't1' in filename:
                entry.sequence_type = 'T1'
            elif 'FLAIR' in filename or 'flair' in filename or 'Flair' in filename:
                entry.sequence_type = 'FLAIR'

            self.add_entry(entry=entry)

        self.add_shapes(keyword='kspace')

        return self

from pathlib import Path
import h5py
from copy import deepcopy

import torch
import torchvision
import numpy as np

from .datasetcontainer import DatasetContainer
from .datasetentry import DatasetEntry
from .datasetinfo import DatasetInfo

from ..logger import get_logger

class DatasetLoader(torch.utils.data.Dataset):
    """
    An iterable datasetloader for the dataset container to make my life easier
    """

    def __init__(self,
                 datasetcontainer: DatasetContainer,
                 train_transforms: torchvision.transforms = None,
                 truth_transforms: torchvision.transforms = None,
                 open_func: callable = None,
                 img_key: str = 'kspace',
                 dataloader_compat: bool = True):
        """
        Args:
            datasetcontainer: The datasetcontainer that is to be loaded
            train_transforms: Transforms the data is gone through before model input
            train_transforms: Transforms the data is gone through before being ground truths
            open_func: Function to open file
            img_key: potential key for opening the file
            dataloader_compat: make datasetloader compatible with pytorch datasetloader
        """

        self.datasetcontainer = datasetcontainer
        self.train_transforms = train_transforms
        self.truth_transforms = truth_transforms
        self.open_func = open_func
        self.img_key = img_key
        self.dataloader_compat = dataloader_compat

        self.logger = get_logger(name=__name__)

        # Checking if dataloader compatibility is enabled
        if dataloader_compat:
            self.logger.info('--------------------------------------------------------------')
            self.logger.info('torch.utils.data.DataLoader compatibility enabled(default=True), '\
                'the first index in shape is assumed to be the slice/image/sample (N, C, H, W).')

            # Checking if all entries have the shape attribute, if not, try to add them.
            if not datasetcontainer.shapes_given():
                self.logger.info('Image shape must be given in entry, for pytorch '\
                    'torch.utils.data.DataLoader compatibility.')
                self.logger.info('Trying to fetch shapes from dataset...')

                # fetching shapes from image files
                datasetcontainer.add_shapes(open_func=open_func, keyword=img_key)

                # Could not fetch shapes, raise error
                if not datasetcontainer.shapes_given():
                    self.logger.warning('Could not fetch shapes, '\
                        'insert manually, aborting program.')
                    raise AttributeError

                self.logger.info('All shapes fetched from files, will continue.')
            else:
                self.logger.info('All entries have the shape attribute, will continue.')

            # Create a dict that maps image index to file and image in file index
            self._index_to_file_and_image = dict()
            counter = 0
            for i, entry in enumerate(datasetcontainer):
                images = entry.shape[0]
                for j in range(images):
                    self._index_to_file_and_image[counter] = (i, j)
                    counter += 1
            self.logger.info('--------------------------------------------------------------')
        else:
            self.logger.info('Pytorch datasetloader compatibility disabled.\n'\
                'The outputs are therefore (batch, C, H, W)')
            self._index_to_file_and_image = None

    def __len__(self):
        if self.dataloader_compat:
            return len(self._index_to_file_and_image)
        else:
            return len(self.datasetcontainer)

    def __getitem__(self, index):
        if self.dataloader_compat:
            index, image_index = self._index_to_file_and_image[index]  # Fetch image (image_index) from volume (index)
        else:
            index = index  # Index corresponds to a file, not image in files
            image_index = ()  # Fetch all images

        entry = self.datasetcontainer[index]
        suffix = Path(entry.image_path).suffix
        image_object = entry.open(open_func=self.open_func)

        if suffix == '.h5':
            image = image_object[self.img_key][image_index]
        elif suffix in ['.nii', '.gz']:
            image = image_object.get_fdata()[image_index]

        # For reconstruction where the train image is masked and thus have a different transform
        if self.train_transforms is not None:
            train = self.train_transforms(image)
        else:
            train = image

        if self.truth_transforms is not None:
            valid = self.truth_transforms(image)
        else:
            valid = np.copy(image)

        return train, valid

    def __iter__(self):
        self.current_index = 0
        self.max_length = len(self)
        return self

    def __next__(self):
        if not self.current_index < self.max_length:
            raise StopIteration
        item = self[self.current_index]
        self.current_index += 1
        return item




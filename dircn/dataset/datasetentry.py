from typing import Union, Dict
import h5py
import nibabel as nib

from pathlib import Path
from ..logger import get_logger


class DatasetEntry(object):
    """
    A class used to store information about the different training objects
    This class works like a regular dict
    """

    def __init__(self,
                 image_path: Union[str, Path] = None,
                 datasetname: str = None,
                 dataset_type: str = None,
                 sequence_type: str = None,
                 field_strength: float = None,
                 pre_contrast: bool = None,
                 post_contrast: bool = None,
                 multicoil: bool = None,
                 shape: tuple = None):
        """
        Args:
            image_path (str, Path): The path where the data is stored
            datasetname (str): The name of the dataset the data is from
            dataset_type (str): What kind of data the data is
            sequence_type (str): The sequence type for MRI
            field_strength (float): Field strength of the scan
            pre_contrast (bool): Is the scan pre contrast
            post_contrast (bool): Is the scan post contrast
            multicoil (bool): Is this a multicoil scan
            shape (tuple): The shape of the data
        """

        self.logger = get_logger(name=__name__)

        if isinstance(image_path, (Path, str)):
            self.image_path = str(image_path)
            if not Path(image_path).is_file():
                self.logger.info('The path: ' + str(image_path))
                self.logger.info('Is not an existing file, are you sure this is the correct path?')
        else:
            self.image_path = image_path

        self.datasetname = datasetname
        self.dataset_type = dataset_type

        self.sequence_type = sequence_type
        self.field_strength = field_strength

        if not isinstance(pre_contrast, bool) and pre_contrast is not None:
            raise TypeError('The variable pre_contrast ', pre_contrast, ' need to be boolean')

        if not isinstance(multicoil, bool) and multicoil is not None:
            raise TypeError('The variable multicoil ', pre_contrast, ' need to be boolean')

        self.pre_contrast = pre_contrast
        self.post_contrast = post_contrast
        self.multicoil = multicoil
        self.shape = shape
        self.score = dict()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def open(self, open_func=None):
        """
        Open the file
        Args:
            open_func (the function to open the file)
        returns:
            the opened file
        """
        if open_func is not None:
            image = open_func(self.image_path)
        else:
            suffix = Path(self.image_path).suffix
            if suffix == '.h5':
                image = self.open_hdf5(self.image_path)
            elif suffix in ['.nii', '.gz']:
                image = self.open_nifti(self.image_path)
            else:
                raise TypeError('cannot open file: ', self.image_path)

        return image

    def open_hdf5(self, image_path):
        return h5py.File(image_path, 'r')

    def open_nifti(self, image_path):
        return nib.load(image_path)

    def add_score(self, img_slice: int, score: Dict[str, float]):
        """
        Add reconstruction score to entry for a given slice in volume
        Args:
            img_slice (int): The slice the score is for (-1 is the entire volume)
            score (Dict[str, float]): Dict of metrics with score
        """
        assert self.shape is not None, 'shape must be added for score support'
        assert isinstance(img_slice, int) and isinstance(score, dict),\
            'img_slice must be int, and score must be dict'
        assert img_slice < self.shape[0], 'img_slice cannot be larger than maximum slice number'

        if img_slice in self.score.keys():
            self.logger.info('there already exists score for this slice, they are overwritten')
        self.score[img_slice] = score

    def add_shape(self, open_func=None, shape=None, keyword='kspace'):
        """
        Add shape to entry
        Args:
            open_func (callable): function for opening file
            shape (tuple): shape of file
            keyword (str): potential keyword for opening file
        """
        if isinstance(shape, tuple):
            self.shape = shape
        else:
            img = self.open(open_func=open_func)
            try:
                shape = img[keyword].shape
            except:
                shape = img.shape

            self.shape = shape

    def keys(self):
        """
        dict keys of class
        """
        return self.to_dict().keys()

    def to_dict(self) -> dict:
        """
        returns:
            dict format of this class
        """
        return {'image_path': self.image_path,
                'datasetname': self.datasetname,
                'dataset_type': self.dataset_type,
                'sequence_type': self.sequence_type,
                'field_strength': self.field_strength,
                'pre_contrast': self.pre_contrast,
                'post_contrast': self.post_contrast,
                'multicoil': self.multicoil,
                'shape': self.shape,
                'score': self.score}

    def from_dict(self, in_dict: dict):
        """
        Args:
            in_dict: dict, dict format of this class
        Fills in the variables from the dict
        """
        if isinstance(in_dict, dict):
            self.image_path = in_dict['image_path']
            self.datasetname = in_dict['datasetname']
            self.dataset_type = in_dict['dataset_type']
            self.sequence_type = in_dict['sequence_type']
            self.field_strength = in_dict['field_strength']
            self.pre_contrast = in_dict['pre_contrast']
            self.post_contrast = in_dict['post_contrast']
            self.multicoil = in_dict['multicoil']
            self.shape = in_dict['shape']
            self.score = in_dict['score']

        return self

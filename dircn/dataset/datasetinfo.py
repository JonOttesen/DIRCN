import json

class DatasetInfo(object):
    """
    Information about the data
    """

    DATASET_TYPES = ['train', 'validation', 'test']  # Predefined types

    def __init__(self,
                 datasetname: str = None,
                 dataset_type: str = None,
                 source: str = None,
                 dataset_description: str = None):

        self._datasetname = datasetname
        self._dataset_type = dataset_type
        self._source = source
        self._dataset_description = dataset_description

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()

    @property
    def datasetname(self):
        """
        Datasetname property
        """
        return self._datasetname

    @datasetname.setter
    def datasetname(self, value):
        self._datasetname = value

    @property
    def source(self):
        """
        Source property
        """
        return self._source

    @source.setter
    def source(self, value):
        self._source = value

    @property
    def dataset_type(self):
        """
        Usage property i.e reconstruction, segmentation etc
        """
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, value):
        self._dataset_type = value

    @property
    def dataset_description(self):
        """
        Usage property i.e reconstruction, segmentation etc
        """
        return self._dataset_description

    @dataset_description.setter
    def dataset_description(self, value):
        self._dataset_type = value

    def keys(self):
        return self.to_dict().keys()

    def to_dict(self) -> dict:
        """
        returns:
            dict format of this class
        """
        return {'datasetname': self.datasetname,
                'dataset_type': self.dataset_type,
                'source': self.source,
                'dataset_description': self.dataset_description}

    def from_dict(self, in_dict: dict):
        """
        Args:
            in_dict: dict, dict format of this class
        Fills in the variables from the dict
        """
        if isinstance(in_dict, dict):
            self.datasetname = in_dict['datasetname']
            self.dataset_type = in_dict['dataset_type']
            self.source = in_dict['source']
            self.dataset_description = in_dict['dataset_description']

        return self


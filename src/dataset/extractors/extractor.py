from abc import abstractmethod
import os.path as osp

from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy


class Extractor:
    """
    Abstract base class for extracting data from a given file path using a specified normalization strategy.
    Attributes:
        file_path (str): Path to the file from which data will be extracted.
        normalization_strategy (NormalizationStrategy): Strategy to normalize the data during extraction.
    """

    def __init__(self, file_path: str, normalization_strategy=NormalizationStrategy.NO_NORMALIZATION):
        """
        Initializes the Extractor with a file path and a normalization strategy.
        Args:
            file_path (str): Path to the file for data extraction.
            normalization_strategy (NormalizationStrategy): Normalization strategy to be used for data extraction.
        """
        self._file_path = file_path
        self._normalization_strategy = normalization_strategy

    @abstractmethod
    def extract(self, number_of_points_per_mesh_entity: int, augment: bool = False):
        """
        Abstract method to extract data from the file. Must be implemented by subclasses.
        Args:
            number_of_points_per_mesh_entity (int): The number of points per mesh entity to extract.
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    @property
    def file_path(self):
        """str: Returns the file path for data extraction."""
        return self._file_path

    @property
    def ifc_class(self):
        """
        str: Returns the IFC class derived from the file name, assumed to be the second segment of the filename separated by '_'.
        """
        return osp.basename(self._file_path).split('_')[1].split('.')[0].lower()

    @property
    def normalization_strategy(self):
        """NormalizationStrategy: Returns the normalization strategy used for extraction."""
        return self._normalization_strategy

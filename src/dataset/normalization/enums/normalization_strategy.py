from enum import Enum

class NormalizationStrategy(Enum):
    """
    Enumeration for defining different types of normalization strategies for mesh data.
    Members:
        ZERO_TO_ONE: Normalizes the mesh vertices such that all coordinates are scaled to fall between 0 and 1.
        MINUS_ONE_TO_ONE: Normalizes the mesh vertices such that all coordinates are scaled to fall between -1 and 1.
        NO_NORMALIZATION: Applies no normalization to the mesh vertices, leaving them in their original state.
    """
    ZERO_TO_ONE = 'zero_to_one'
    MINUS_ONE_TO_ONE = 'minus_one_to_one'
    NO_NORMALIZATION = 'no_normalization'

from enum import Enum


class NormalizationStrategy(Enum):

    ZERO_TO_ONE = 'zero_to_one'
    MINUS_ONE_TO_ONE = 'minus_one_to_one'
    NO_NORMALIZATION = 'no_normalization'
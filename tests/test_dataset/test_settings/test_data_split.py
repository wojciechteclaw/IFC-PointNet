import pytest
from src.dataset.settings.data_split import DataSplit

def test_initialization_with_default_ratios():
    data_split = DataSplit()
    assert data_split.training_ratio == 0.8
    assert data_split.testing_ratio == 0.1
    assert data_split.validation_ratio == 0.1

def test_initialization_with_custom_ratios():
    data_split = DataSplit(training_split_ratio=0.7, testing_split_ratio=0.2, validation_split_ratio=0.1)
    assert data_split.training_ratio == 0.7
    assert data_split.testing_ratio == 0.2
    assert data_split.validation_ratio == 0.1

def test_initialization_with_incorrect_ratios():
    with pytest.raises(AssertionError, match="The sum of the ratios must be 1.0"):
        DataSplit(training_split_ratio=0.5, testing_split_ratio=0.3, validation_split_ratio=0.3)

def test_initialization_with_negative_ratios():
    with pytest.raises(AssertionError, match="The ratios must be positive"):
        DataSplit(training_split_ratio=-0.1, testing_split_ratio=0.6, validation_split_ratio=0.5)

def test_initialization_with_ratios_sum_not_one():
    with pytest.raises(AssertionError, match="The sum of the ratios must be 1.0"):
        DataSplit(training_split_ratio=0.4, testing_split_ratio=0.4, validation_split_ratio=0.3)

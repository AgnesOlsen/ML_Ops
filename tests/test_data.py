import torch
import pytest
import os
from src.ml_ops_ex.data import corrupt_mnist


@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt"), reason="Data files not found")
def test_data():
    """Test the dimensions and labels of the processed MNIST data."""
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Train Dataset did not have the correct number of samples"
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all()


"""def test_my_dataset():
    Test the MyDataset class.
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)"""

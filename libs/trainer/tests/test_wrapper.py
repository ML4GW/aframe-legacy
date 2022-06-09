import sys

import numpy as np
import pytest

from bbhnet.data.transforms.transform import Transform
from bbhnet.trainer.wrapper import trainify


@pytest.fixture(params=[True, False])
def validate(request):
    return request.param


@pytest.fixture(params=[True, False])
def preprocess(request):
    return request.param


def dataset(batches):
    for i in range(batches):
        x = np.random.randn(8, 2, 512)
        y = np.random.randint(0, 2, size=(8, 1))
        yield x, y


class Preprocessor(Transform):
    def __init__(self):
        super().__init__()
        self.factor = self.add_parameter(2)

    def forward(self, x):
        return self.factor * x


@pytest.fixture
def get_data(validate, preprocess):
    def fn(batches: int):
        train_dataset = dataset(batches)
        valid_dataset = dataset(batches) if validate else None
        preprocessor = Preprocessor if preprocess else None
        return train_dataset, valid_dataset, preprocessor

    return fn


@pytest.fixture(params=[True, False])
def data_fn(request, get_data):
    # make sure we can have functions that overlap their args
    if request.params:

        def fn(batches: int, max_epochs: int):
            return get_data(batches)

    else:

        def fn(batches: int):
            return get_data(batches)

    return fn


def test_wrapper(data_fn, preprocess):
    fn = trainify(data_fn)

    # make sure we can run the function as-is with regular arguments
    train_dataset, valid_dataset = fn(4)
    for i, (X, y) in enumerate(train_dataset):
        continue
    assert i == 3

    # call function passing keyword args
    # for train function
    result = fn(
        4,
        max_epochs=1,
        arch="resnet",
        layers=[2, 2, 2],
    )
    assert len(result["train_loss"]) == 1

    sys.argv = [
        None,
        "--batches",
        "4",
        "--max-epochs",
        "1",
        "resnet",
        "--layers",
        "2",
        "2",
    ]

    # since trainify wraps function w/ typeo
    # looks for args from command line
    # i.e. from sys.argv
    result = fn()
    assert len(result["train_loss"]) == 1

    # TODO: check that if preprocess, there's
    # an extra parameter in the model. use a
    # mock in dataset to check that if validate,
    # it gets called twice as many times as
    # expected

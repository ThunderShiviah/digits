import pytest
import numpy as np
import sklearn
from sklearn import utils
import pandas as df
from common import data_io

def test_get_settings_exists():
    assert data_io.get_settings()

def test_get_settings_returns_dict():
    assert type(data_io.get_settings()) == type({})

def test_get_train_returns_something():
    assert data_io.get_train()
   
def test_get_train_returns_consistent_lengths():
    try:
        X, y = data_io.get_train() # This will fail if get_train only returns X.
        assert utils.check_X_y(X, y)
    except Exception as e:
        assert False,e
        pass

def test_get_train_can_return_pandas_df():
    X, y = data_io.get_train(as_df=True)
    assert isinstance(X, df.DataFrame)

def test_get_train_can_return_pandas_df_unparsed():
    X, = data_io.get_train(as_df=True, parsed=False)
    assert isinstance(X, df.DataFrame)


def test_get_train_can_return_numpy_array():
    X, y = data_io.get_train(as_df=False)
    assert isinstance(X, type(np.array(1)))
    assert isinstance(y, type(np.array(1)))

def test_get_train_can_return_numpy_array_unparsed():
    X, = data_io.get_train(as_df=False, parsed=False)
    assert isinstance(X, type(np.array(1)))

def test_get_test_returns_something():
    assert data_io.get_test()

def test_get_test_can_return_pandas_df():
    y_test, = data_io.get_test(as_df=True)
    assert isinstance(y_test, df.DataFrame)


def test_get_test_can_return_numpy_array():
    y_test, = data_io.get_test(as_df=False)
    assert isinstance(y_test, type(np.array(1)))

def test_load_model_returns_something():
    assert data_io.load_model()

def test_load_model_returns_pipeline():
    assert isinstance(data_io.load_model(), sklearn.pipeline.Pipeline)

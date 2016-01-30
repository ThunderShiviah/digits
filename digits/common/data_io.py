"""
This script provides a data API between outside data sources (as specified
in SETTINGS.json) and the main scripts.

TODO: Refactor get_train and get_test into common base class."""

import os
import json
from sklearn import utils

def get_settings():
    """Returns SETTINGS.json as dict."""
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def get_train(as_pd=True, parsed=True, headers=False):
    """Returns training data as tuple.

    --------------------------------
    args:None
    returns: (X_train [, y_train])"""

    assert utils.check_X_y(X_train, y_train)

    return X_train, y_train


def get_test(as_pd=True, parsed=True, headers=False):
    """Returns test data as a tuple (singlet).

    --------------------------------
    args:None
    returns: (X_test,)"""

    pass


def save_model(model):
    """Serializes model to disk. 

    Currently saves model as a pickle.

    -----------------------------------
    args: scikit compatible model
    returns: None # Should this return something?
    """

    pass

def load_model():
    """Returns model from location specified by SETTINGS.json.

    ---------------------------------
    args:None
    returns: scikit compatible model"""
    pass

def write_submission(predictions, as_df=True): # Do I really need the as_df param?
    """Writes submission to location specified by SETTINGS.json.

    ----------------------------------
    args: predictions
    params: as_df - if True, processes predictions as pandas dataframe."""

    pass



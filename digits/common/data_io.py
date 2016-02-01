"""
This script provides a data API between outside data sources (as specified
in SETTINGS.json) and the main scripts.

TODO: Refactor get_train and get_test into common base class."""

import os
import json
from sklearn import utils
import pandas as pd

def get_settings():
    """Returns SETTINGS.json as dict."""
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def get_data(data_path, as_df=True, parsed=True):
    """Returns training data as tuple.

    --------------------------------
    args 
    ----
    as_df: Bool
    If True, returns data as pandas DataFrame.

    parsed: Bool
    If True, attempts to import correct data parsing function from 'parse_import_path' key set in SETTINGS.json.
    'parse_import_path' value should be formatted as 'module_name.parse_function_name'.

    returns: (X_train [, y_train])"""

    path = get_settings()[data_path]
    

    data = pd.read_csv(path)

    if parsed and data_path=="train_data_path":
        """Try to get the import path of the data parsing script from the 'parse_import_path' key value in SETTINGS.json. Use the associated data parsing function to parse the data into an X_train and y_train data set. Use sklearn.utils.check_X_y to validate the X_train and y_train for use in sklearn models."""
        import importlib
        try:
            parse_import_path = get_settings()["parse_import_path"]
        except KeyError as err:
             print("{bad_key} key missing from SETTINGS.json.".format(bad_key=err))
             raise KeyError("{bad_key} key missing from SETTINGS.json.".format(bad_key=err))
        module_name, func_name = parse_import_path.rsplit('.', 1) 
        try:
            _parse = getattr(importlib.import_module(module_name),func_name)
        except ImportError as err:
            raise ImportError("'parse_import_path' value in SETTINGS.json not valid. Import failed with message: ", err.args) # TODO: Clean up error formatting.

        X_train, y_train = _parse(data)
        assert utils.check_X_y(X_train, y_train)

        if not as_df: # Return as numpy array
            return X_train.values, y_train.values

        return X_train, y_train

    else:
        if not as_df:
            data = data.values

        return (data,) # Should return a tuple


        
def get_train(as_df=True, parsed=True):
    """Returns training data as specified by 'train_data_path'
    
    Wrapper around get_data function."""
    return get_data("train_data_path", as_df=as_df, parsed=parsed)

def get_test(as_df=True): # TODO: implement a parsed setting.
    """Returns test data as specified by 'test_data_path'
    
    Wrapper around get_data function."""

    return get_data("test_data_path", as_df=as_df, parsed=False)

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



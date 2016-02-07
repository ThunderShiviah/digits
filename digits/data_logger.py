"""data_logger.py is a module for logging experimental machine learning runs including model parameters and scores."""
import json
from common import data_io
from time import strftime
from pprint import pprint

def log(clf, score):
    log_path = data_io.get_settings()["log_path"]
    date = strftime("%d-%m-%Y-%H-%M-%S")
    log_file = log_path + date + ".json" 

    log_obj = {"date":date,
            "classifier":clf.get_params(),
            "score": score}


    with open(log_file, "w") as fp:
        json.dump(log_obj, fp, 
                sort_keys = True, indent = 4,
                ensure_ascii=False)


from collections import OrderedDict

def make_log(clf, verbose=True):
    
    X_train, X_test, y_train, y_test = data_io.get_train_test_split() 
    clf.fit(X_train, y_train)

    log = OrderedDict({
        "timestamp": strftime("%d-%m-%Y-%H-%M-%S"),
        "model_name": clf.__str__(),
        "model_params":clf.get_params(),
        "training_data":data_io.get_settings()["train_data_path"],
        "testing_data":data_io.get_settings()["test_data_path"],
        "score": clf.score(X_test, y_test),
        "model_file_handle": data_io.save_model(clf),
            })

    if verbose:
        pprint(log)
    
    return log

if __name__ == "__main__":
    log(1,1)

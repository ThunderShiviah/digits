"""data_logger.py is a module for logging experimental machine learning runs including model parameters and scores."""
import json
from common import data_io
from time import strftime

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

if __name__ == "__main__":
    log(1,1)

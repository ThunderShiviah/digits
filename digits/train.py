from __future__ import print_function
import data_io
import logging
import cv
from features import FeatureMapper, SimpleTransform
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autosklearn import classification

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s')

def feature_extractor():

    features = [('StandardScalar', 'FullDescription', StandardScaler()),]
    combined = FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [
            #("extract_features", features),
            ("RandomForest", RandomForestClassifier(n_estimators=50, 
                verbose=5,
                n_jobs=1,
                min_samples_split=30,
                random_state=3465343)),
            #("StandardScaler", StandardScaler()),
            #("SVM", SVC(max_iter = 5,
            #    random_state=3465343)),
            ]
    return Pipeline(steps)

def main():
    print("Reading in the training data")
    train = data_io.get_train_df()

    print("Extracting features and training model")
    X_train = train[[x for x in train.columns if x != 'label']]
    y_train = train['label']

    print("Loading the automl classifier")
    #automl = classification.AutoSklearnClassifier()
    print("training the classifier")
    #automl.fit(X_train.values, y_train.values)

    automl = classification.AutoSklearnClassifier(time_left_for_this_task=15,
            per_run_time_limit=5,
            tmp_folder='/tmp/autoslearn_example_tmp',
            output_folder='/tmp/autosklearn_example_out')
    automl.fit(X_train.values, y_train.values, dataset_name='digits')
    #print(automl.score(X_test, y_test))

    print("Saving the classifier")
    data_io.save_model(automl)

if __name__=="__main__":
    main()

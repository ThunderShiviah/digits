from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline

def get_pipeline():
    pipeline = make_pipeline(DummyClassifier(strategy='stratified', random_state=0),
        )
    return pipeline



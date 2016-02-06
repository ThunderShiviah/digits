from common import data_io
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import make_pipeline

X, y= data_io.get_train()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipeline = make_pipeline(DummyClassifier(strategy='stratified', random_state=0),
        )


if __name__ == "__main__":

    from data_logger import log

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(pipeline)
    print(score)
    print(pipeline.get_params())

    data_io.save_model(pipeline)
    model = data_io.load_model()
    print(model)


    #log(pipeline, score)


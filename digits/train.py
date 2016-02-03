from common import data_io
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split

X, y= data_io.get_train()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DummyClassifier(strategy='most_frequent', random_state=0)


if __name__ == "__main__":

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)

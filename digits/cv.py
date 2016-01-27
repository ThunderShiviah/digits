import data_io
from sklearn import cross_validation as cv
from sklearn import metrics

def get_score_report(classifier, expected, predicted):
    print("Classification report for classifier %s:\n%s\n"
                  % (classifier, metrics.classification_report(expected, predicted)))

def get_confusion_matrix(expected, predicted):
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

def get_cv_predict(clf, data, target):
    return cv.cross_val_predict(clf, data)

def get_cv_score():

    classifier = data_io.load_model()
    train = data_io.get_train_df()
    scores = cv.cross_val_score(classifier, train[[x for x in train.columns if x != 'label']], train['label'])

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
    get_cv_score()

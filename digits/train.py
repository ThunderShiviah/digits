import data_io
import cv
from features import FeatureMapper, SimpleTransform
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


def feature_extractor():
    features = [('FullDescription-Bag of Words', 'FullDescription', CountVectorizer(max_features=100)),]
    combined = FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [
             #("extract_features", features),
             ("classify", RandomForestClassifier(n_estimators=3, 
                                                verbose=2,
                                                n_jobs=1,
                                                min_samples_split=30,
                                                random_state=3465343))]
    return Pipeline(steps)

def main():
    print("Reading in the training data")
    train = data_io.get_train_df()

    print("Extracting features and training model")
    classifier = get_pipeline()
    data = train[[x for x in train.columns if x != 'label']]
    target = train['label']
    classifier.fit(data, target )
    #predictions = cv.get_cv_predict(classifier, data, target)

    expected, predicted = target, classifier.predict(data)
    
    cv.get_score_report(classifier, expected, predicted)
    cv.get_confusion_matrix(expected, predicted)
    

    print("Saving the classifier")
    data_io.save_model(classifier)
    
if __name__=="__main__":
    main()

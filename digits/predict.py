import data_io
import pickle
import numpy as np
import cv

def main():
    print("Loading the classifier")
    classifier = data_io.load_model()
    
    print("Making predictions") 
    valid = data_io.get_valid_df()
    predictions = classifier.predict(valid)   
    predictions = np.rint(predictions) # Round predictions to nearest integer.

    print("Writing predictions to file")
    data_io.write_submission(predictions)

if __name__=="__main__":
    main()

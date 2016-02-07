from common import data_io
from data_logger import make_log
from pipeline import get_pipeline

if __name__ == "__main__":

    model = get_pipeline()

    print("--------------------Trying logging util----------")
    make_log(model)



    #log(pipeline, score)


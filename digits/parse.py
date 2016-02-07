"""This is a dataset specific file that parses data into an X_train and y_train set for use in sklearn models."""

def parse_data(data):
    return data[[x for x in data.columns if x != 'label']], data['label']

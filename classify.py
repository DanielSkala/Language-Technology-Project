from sklearn.metrics import classification_report, multilabel_confusion_matrix
import pickle
import numpy as np
import pandas as pd
from train_classifier import Dataset


class Classifier:
    def __init__(self):
        self.model = pickle.load(open("model.p", "rb"))

    def format_as_dataset(self, text):
        textdf = pd.DataFrame(columns=["Argument ID", "Merged"])
        textdf.loc[0] = [1,text]
        return Dataset(textdf)

    def classify(self, text):
        textdf = pd.DataFrame(columns=["Argument ID", "Merged"])
        textdf.loc[0] = [1,text]
        textdata = Dataset(textdf)

        prediction = self.model.predict(textdata.data)
        return prediction


if __name__ == '__main__':
    classifier = Classifier()
    classifier.classify("We should end racial profiling racial profiling is a preconceived idea of people that views an entire race as criminal")
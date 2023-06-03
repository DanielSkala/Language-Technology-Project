from typing import Literal, List, Optional
import pandas as pd
from model_onecol import get_model
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import pickle

class Dataset:
    """
    Dataset object containing data and optionally labels.
    """

    data: pd.DataFrame
    data_ids: pd.Series
    labels: Optional[pd.DataFrame]
    label_names: Optional[List[str]]
    label_ids: Optional[pd.Series]

    def __init__(self, data: pd.DataFrame, labels: Optional[pd.DataFrame] = None):
        self.data = data.drop('Argument ID', axis=1)
        self.data_ids = data['Argument ID']
        if labels is not None:
            self.labels = labels.drop('Argument ID', axis=1)
            self.label_names = self.labels.columns.to_list()
            self.label_ids = labels['Argument ID']

    @property
    def has_labels(self):
        """
        Indicates whether this dataset has labels or not
        """
        return self.labels is not None

def load_dataset(dataset_type: Literal['training', 'validation', 'test']) -> Dataset:
    """
    Load a dataset
    """
    arguments = pd.read_csv(f'datasets/arguments-{dataset_type}.tsv', sep='\t')
    labels = pd.read_csv(f'datasets/labels-{dataset_type}.tsv', sep='\t')

    arguments["Merged"] = arguments["Premise"] + arguments["Conclusion"]
    arguments = arguments.loc[:, arguments.columns!='Stance']
    arguments = arguments.loc[:, arguments.columns!='Premise']
    arguments = arguments.loc[:, arguments.columns!='Conclusion']


    return Dataset(
        arguments,
        labels
            if dataset_type != 'test'
            else None
        )

def main():
    """
    Run the program
    """
    train_dataset = load_dataset('training')
    model = get_model().fit(train_dataset.data, train_dataset.labels)


    validation_dataset = load_dataset('validation')

    z = validation_dataset.data.merge(validation_dataset.labels, left_index=True, right_index=True)
    z.to_csv("datasets/validation_merged.csv")

    validation_predictions = model.predict(validation_dataset.data)
    print(classification_report(
        validation_dataset.labels,
        validation_predictions,
        target_names=validation_dataset.label_names
        ))
    
    pickle.dump(model, open("model.p", "wb"))

    print(validation_dataset.data)


if __name__ == '__main__':
    main()
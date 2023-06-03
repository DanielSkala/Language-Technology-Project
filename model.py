import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier

def binarize(stance: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({'stance': stance == 'in favor of'})

def get_model():
    return Pipeline([
        (
            'features',
            ColumnTransformer([
                ('conclusion_vector', HashingVectorizer(), 'Conclusion'),
                ('premise_vector', HashingVectorizer(), 'Premise'),
                ('stance', FunctionTransformer(binarize), 'Stance')
            ])
        ),
        ('mlsvc', MultiOutputClassifier(LinearSVC()))
    ])
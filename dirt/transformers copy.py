import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dirt_pred_preprocess import custom_preprocess

class DataLoader(BaseEstimator, TransformerMixin):
    def __init__(self, path):
        self.path = path

    def pickle_load(self, path):
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()
        
        return data

    def load_data(self):
        data = self.pickle_load(self.path)

        return data
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.load_data()

        return X

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, preprocess_flag):
        self.preprocess_flag = preprocess_flag
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.preprocess_flag:
            try:
                X = custom_preprocess(X)
            except:
                print("Error related with custom preprocessing file")
        
        return X
    
class FeatureTargetSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, target):
        self.target = target

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        print("tranasformer")
        print(X)
        # X = X.drop(columns=[self.target])
        # y = X[self.target]

        # return X, y
    
class FeatureDataTypeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, target):
        self.target = target

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        FEATURES = X.columns.drop(self.target)
        NUMERICAL = X[FEATURES].select_dtypes('number').columns
        CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))

        return X, (NUMERICAL, CATEGORICAL)
    


# split features and target



# y = LabelEncoder().fit_transform(y)

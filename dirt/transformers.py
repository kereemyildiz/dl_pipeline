import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pickle 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from preprocess import custom_preprocess
from keras.utils import np_utils

class DataLoader(BaseEstimator, TransformerMixin):
    def __init__(self, path, load_type):
        self.path = path
        self.load_type = load_type


    def pickle_load(self, path):
        file = open(path, 'rb')
        data = pickle.load(file)
        file.close()
        
        return data
     
    def csv_load(self, path):
        data = pd.read_csv(f"{path}")
        return data

    def load_data(self):
        if self.load_type == "pickle":
            data = self.pickle_load(self.path)
        elif self.load_type == "csv":
            data = self.csv_load(self.path)

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
        y = X[self.target]
        X = X.drop(columns=[self.target])

        return X, y
    

class FeatureDataTypeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X, y = X
        FEATURES = X.columns
        NUMERICAL = X[FEATURES].select_dtypes('number').columns
        CATEGORICAL = pd.Index(np.setdiff1d(FEATURES, NUMERICAL))

        return X, y, (NUMERICAL, CATEGORICAL)
    

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X, y, _ = X
        classification_type = ""
        if y.nunique() > 2: # one-hot encode (multi-class classification)
            y = np_utils.to_categorical(y)
            classification_type = "multi"

        else: # binary classification
            y = LabelEncoder().fit_transform(y)
            classification_type = "binary"

        return X, y, _, classification_type
        

# split features and target



# y = LabelEncoder().fit_transform(y)

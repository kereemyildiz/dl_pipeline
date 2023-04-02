
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def custom_preprocess(seq):
    drop_col = "Test_no,Weight,Weight_categorical,Water_level_10s,Water_level_20s,Water_level_30s,WL_exactpoint_60s,WL_exactpoint_30s".split(',')
    seq.drop(columns=drop_col, inplace=True)
    return seq
            

def preprocess(data):
    # Dropping constant columns
    drop_cols = [e for e in data.columns if data[e].nunique() == 1]
    X = data.drop(drop_cols, axis=1)
    scaler = StandardScaler().fit(X)
    scaled_X = scaler.transform(X)
    return scaled_X


def find_non_numeric_cols(X):
    cols = X.columns
    num_cols = X._get_numeric_data().columns
    non_numerics = list(set(cols) - set(num_cols))
    return non_numerics


def Encode(X, y, non_numeric_cols):
    non_numeric_cols = find_non_numeric_cols(X)
    encoded_X = pd.get_dummies(data = X, columns = non_numeric_cols)
    encoded_Y  = LabelEncoder().fit_transform(y)
    
    return encoded_X, encoded_Y


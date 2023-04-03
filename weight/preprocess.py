
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def custom_preprocess(seq):
    drop_columns = ["No", "Yük Tipi"]
    seq.drop(columns=drop_columns, inplace=True)
    seq["Yük miktarı (kg)"] = seq["Yük miktarı (kg)"].apply(weight_categorize)

    return seq

def weight_categorize(weight):
    
    if weight < 3.5:
        return 0 # çeyrek

    elif weight > 6.5:
        return 2 # tam
    
    else:
        return 1 # yarım
    
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


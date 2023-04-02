
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def custom_preprocess(seq):
    data = []
    drop_list = ["3_3","6_1","9_2","24","30","41","42","23","26"]
    for i in range(len(seq)):

        if str(seq[i][1].name) in drop_list:
            continue
        else:
            data.append(seq[i])

    data = series_to_dataframe(data)

    return data
            
def series_to_dataframe(data):

    df = pd.DataFrame()
    weight, min12, min14, min16, sas, dirt = [], [], [], [], [], []

    for i in range(len(data)):

        min12.append(data[i][1]["12_"])
        min14.append(data[i][1]["14_"])
        min16.append(data[i][1]["16_"])
        weight.append(data[i][1]["Yük Seviyesi"])
        sas.append(data[i][1]["SY (Toplam)\nSAS"])
        dirt.append(data[i][1]["Kir 2.0"])

    df.insert(0, "12_", min12)
    df.insert(1, "14_", min14)
    df.insert(2, "16_", min16)
    df.insert(3, "Weight", weight)
    df.insert(4, "SAS", sas)
    df.insert(5, "Dirt", dirt)

    return df

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


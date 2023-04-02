import pickle
import pandas as pd
import os

def load_data():
    dataset_path = "../../Data/"
    dataset_version = "D1_turbidity_data_2022_08_16/"
    used_data = "processed/sivi/"
    data = "sivi_combined_tests_1sec_pickle"
    data_path = dataset_path + dataset_version + used_data + data

    data = pickle_load(data_path)

    return data

def pickle_load(data_path):
    file = open(data_path, 'rb')
    data = pickle.load(file)
    file.close()
    
    return data

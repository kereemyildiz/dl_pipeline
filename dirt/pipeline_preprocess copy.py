from sklearn.pipeline import Pipeline
from transformers import DataLoader, CustomPreprocessor, FeatureTargetSplitter, FeatureDataTypeExtractor
import pandas as pd
from sklearn.preprocessing import LabelEncoder


DATA_PATH = "../../Data/D1_turbidity_data_2022_08_16/processed/sivi/sivi_combined_tests_1sec_pickle"
CUSTOM_PREPROCESS = True
TARGET = 'Dirt'

# load data and then custom preprocessing if necessary
preprocessor_pipeline = Pipeline([
    ('load_data', DataLoader(DATA_PATH)),
    ('custom_preprocess', CustomPreprocessor(CUSTOM_PREPROCESS)),
    ('feature_type_extract', FeatureDataTypeExtractor(TARGET)),
    #('feauture_target_split', FeatureTargetSplitter(TARGET))

])


df, FEATURES = preprocessor_pipeline.fit_transform(None) 

NUMERICAL = FEATURES[0]
CATEGORICAL = FEATURES[1]

print(NUMERICAL, CATEGORICAL, type(df))

# split features and target

# X = df.drop(columns=[TARGET])
# y = df[TARGET]

# y = LabelEncoder().fit_transform(y)


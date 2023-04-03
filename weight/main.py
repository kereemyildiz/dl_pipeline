import numpy as np
from sklearn.pipeline import Pipeline
from transformers import DataLoader, CustomPreprocessor, FeatureTargetSplitter, FeatureDataTypeExtractor
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from models import baseline_model, custom_model, ensemble_model, autokeras_model
import tensorflow as tf
from performance import calculate_roc_auc
import matplotlib.pyplot as plt

# you should specify necessary informations about data
SEED = config.SEED
DEBUG_PATH = config.DEBUG_PATH
DATA_PATH = config.DATA_PATH
CUSTOM_PREPROCESS = config.CUSTOM_PREPROCESS
TARGET = config.TARGET
LOAD_TYPE = config.LOAD_TYPE

# load data and then custom preprocessing if necessary
preprocessor_pipeline = Pipeline([
    ('load_data', DataLoader(DATA_PATH)),
    ('custom_preprocess', CustomPreprocessor(CUSTOM_PREPROCESS)),
    ('feauture_target_split', FeatureTargetSplitter(TARGET)),
    ('feature_type_extract', FeatureDataTypeExtractor(TARGET))

])

X, y, FEATURES = preprocessor_pipeline.fit_transform(None) 

NUMERICAL = FEATURES[0]
CATEGORICAL = FEATURES[1]

y = LabelEncoder().fit_transform(y)


# TODO: ABOVE THIS LINE SHOULD MOVE TO DATASET.PY WHICH IS IN DEVELOP BRANCH


# Define categorical pipeline
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

# Define numerical pipeline
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Combine categorical and numerical pipelines
preprocessor = ColumnTransformer([
    ('cat', cat_pipe, CATEGORICAL),
    ('num', num_pipe, NUMERICAL)
])

# Fit a pipeline with transformers and an estimator to the training data
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cvscores_train = []
cvscores_test = []

for train, test in kfold.split(X, y):

    #model = ensemble_model(number_of_models=5) # in order to surpass tf warning message, move this line outside of the loop
    model = baseline_model()
 # create model
    pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', KerasClassifier(model, epochs=50, batch_size=5, verbose=1)),
    ])

    x_train, x_test = pd.DataFrame(X.iloc[train]), pd.DataFrame(X.iloc[test])
    y_train, y_test = np.asarray(pd.DataFrame(y).iloc[train]), np.asarray(pd.DataFrame(y).iloc[test])

    pipe.fit(x_train, y_train)

    #  evaluate the model
    y_train_pred = pipe.predict(x_train)
    y_test_pred = pipe.predict(x_test)


    score_train = accuracy_score(y_train, y_train_pred.round())
    score_test = accuracy_score(y_test, y_test_pred.round())

    print('Train Accuracy: %.3f' % score_train ) # format float output
    print('Test Accuracy: %.3f' % score_test ) # format float output

    cvscores_train.append(score_train * 100)
    cvscores_test.append(score_test * 100)
 
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_train), np.std(cvscores_train)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_test), np.std(cvscores_test)))

# plot of train and test scores vs tree depth
plt_idxs = np.arange(len(cvscores_train))
plt.plot(plt_idxs, cvscores_train, '-o', label='Train')
plt.plot(plt_idxs, cvscores_test, '-o', label='Test')
plt.legend()
plt.show()

# y_pred = clf1.predict(X_test) 
# print('Test Accuracy: %.3f' % metrics.accuracy_score(y_test, y_pred)) # format float output


# print(f"Train ROC-AUC: {calculate_roc_auc(pipe, X_train, y_train):.4f}")
# print(f"Test ROC-AUC: {calculate_roc_auc(pipe, X_test, y_test):.4f}")


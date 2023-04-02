from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, concatenate, Dropout

def baseline_model():
    model = Sequential()
    model.add(Dense(64, input_shape=(7,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def baseline_ensemble_model():
    return Sequential(
        [BatchNormalization(input_shape=(7,)),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),])

def custom_model():
    model = Sequential()
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def ensemble_model(number_of_models, models=None):
    if type(models) is not list: # models not specified, create model function passed 
        models = [baseline_ensemble_model() for i in range(number_of_models)]
    for i, model in enumerate(models):
        for layer in model.layers:
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name

    ensemble_inputs = [model.input for model in models]
    ensemble_outputs = [model.output for model in models]

    merge = concatenate(ensemble_outputs)
    batchnorm = BatchNormalization()(merge)
    hidden = Dense(8, activation='relu')(batchnorm)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=ensemble_inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def autokeras_model():
    pass
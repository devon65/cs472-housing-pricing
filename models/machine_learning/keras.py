from data.loader import HousingDataset as hd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

def create_model(inputDim=12):
    # create model
    model = Sequential()
    model.add(Dense(40, input_dim=inputDim, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, activation='relu'))

    # because our labels are very large, if the learning rate is much bigger we only see nan's in the metrics
    sgd = optimizers.SGD(lr=.001, decay=1e-34, momentum=.9, nesterov=True)
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae','acc'])
    return model



def is_x_years(d, x=5):
    _, (start_year, end_year), _, _ = d
    if end_year - start_year + 1 == x:
        return True
    else:
        return False

def prep_data(d, feature_columns, target_column):
    city, (start_year, end_year), x, y = d
    x = x[feature_columns]
    y = y[target_column].to_numpy().item()
    return x.to_numpy(), y

def is_stranger(x):
    if len(x) > 1 or np.any(np.isnan(x)):
        return True
    else:
        return False

def getData(code=["43-0000"],window_length=5):
    data = hd(code)
    print("Dataset loaded.")
    preloaded = [ o for o in data.iterate_areas_with_flat_window(window_length, make_target=True)]

    filtered = filter(lambda d: is_x_years(d, window_length), preloaded)

    col_names = [ "YEAR", "TOT_EMP", "H_MEDIAN" ]
    train_cols = [ c for c in col_names ]

    for i in range(1, window_length-1):
        train_cols += [f"{c}_{i}" for c in col_names]


    prepped = [ prep_data(f, train_cols, "HOUSING_INDEX") for f in filtered ]
    cleaned = filter(lambda x: not is_stranger(x[0]) and not np.isnan(x[1]), prepped)
    inputs, targets = zip(*cleaned)

    inputScaleFactor = np.max(inputs, axis=0)
    inputs = inputs/inputScaleFactor

    targetScaleFactor = np.max(targets, axis=0)
    targets = targets/targetScaleFactor

    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=0.1)
    print(f"Dataset prepped.")
    print(f"Num training instances: {len(train_inputs)}")
    data = [train_inputs, test_inputs, train_targets, test_targets]
    scalingFactors = [inputScaleFactor, targetScaleFactor]
    return data,scalingFactors


def train(data=None, scalingFactors=None):
    if data is None:
        data,scalingFactors = getData()
    train_inputs, test_inputs, train_targets, test_targets = data
    inputScaleFactor, targetScaleFactor = scalingFactors

    train_inputs = np.array(np.concatenate([*train_inputs], axis=0))
    test_inputs = np.array(np.concatenate([*test_inputs], axis=0))
    train_targets = np.array(train_targets).reshape(-1, 1)
    test_targets = np.array(test_targets).reshape(-1, 1)
    assert not np.any(np.isnan(train_inputs))
    assert not np.any(np.isnan(train_targets))

    n_inputs = train_inputs[-1]
    model = KerasRegressor(build_fn=create_model,
                            epochs=1000,
                            batch_size=500,
                            verbose=2,
                        )

    model.fit(train_inputs, train_targets)
    
    acc = model.score(test_inputs, test_targets)
    preds = model.predict(test_inputs)
    rmse_preds = preds*targetScaleFactor
    rmse_targets = test_targets*targetScaleFactor
    diffs = test_targets - preds
    rmse = np.sqrt(mean_squared_error(rmse_targets, rmse_preds))
    return model, acc, rmse, (train_inputs, test_inputs, train_targets, test_targets)

if __name__ == "__main__":
    model, acc, rmse, data = train()
    print(f"Accuracy: {acc}; RMSE: {rmse}")





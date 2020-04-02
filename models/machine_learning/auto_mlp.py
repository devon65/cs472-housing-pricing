from data.loader import HousingDataset as hd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

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

def train():
    window_length = 5
    data = hd(["43-0000"])
    print("Dataset loaded.")
    preloaded = [ o for o in data.iterate_areas_with_flat_window(window_length, make_target=True)]

    filtered = filter(lambda d: is_x_years(d, window_length), preloaded)

    col_names = [ "YEAR", "TOT_EMP", "H_MEDIAN" ]
    train_cols = [ c for c in col_names ]

    for i in range(1, 4):
        train_cols += [f"{c}_{i}" for c in col_names]

    prepped = [ prep_data(f, train_cols, "HOUSING_INDEX") for f in filtered ]
    cleaned = filter(lambda x: not is_stranger(x[0]) and not np.isnan(x[1]), prepped)
    inputs, targets = zip(*cleaned)

    train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=0.1)
    print(f"Dataset prepped.")
    print(f"Num training instances: {len(train_inputs)}")

    train_inputs = np.array(np.concatenate([*train_inputs], axis=0))
    test_inputs = np.array(np.concatenate([*test_inputs], axis=0))
    train_targets = np.array(train_targets).reshape(-1, 1)
    test_targets = np.array(test_targets).reshape(-1, 1)

    model = MLPRegressor(   learning_rate="adaptive", 
                            max_iter=100000,
                            n_iter_no_change=100,
                            verbose=True,
                            solver="adam",
                            tol=1e-6,
                            hidden_layer_sizes=(40, 80, 40)
                        )

    model.fit(train_inputs, train_targets)
    
    acc = model.score(test_inputs, test_targets)
    preds = model.predict(test_inputs)
    diffs = test_targets - preds
    rmse = np.sqrt(mean_squared_error(test_targets, preds))
    return model, acc, rmse, (train_inputs, test_inputs, train_targets, test_targets)

if __name__ == "__main__":
    model, acc, rmse, data = train()
    print(f"Accuracy: {acc}")





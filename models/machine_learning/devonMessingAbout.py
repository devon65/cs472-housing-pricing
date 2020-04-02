import sys
sys.path.append('..')
from data.loader import HousingDataset as hd
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNNRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

ALL_OCCUPATIONS = ["00-0000"]
CS_CODES = ["15-1131", "15-1132", "15-1133", "15-1134"]
HIGHLY_CORRELATED_CODES = []
OCC_CODES = CS_CODES
DESIRED_ATTRS = ['AREA_NAME', 'YEAR', 'PRIM_STATE', "TOT_EMP", "H_MEAN", "A_MEAN", "H_MEDIAN", "A_MEDIAN"]
NOMINAL_ATTRS = ['PRIM_STATE']
STATE_ABREVS = 'PRIM_STATE'
HOUSING_DATA = 'HOUSING_INDEX'

# def make_column_types():
#     column_types = [NOMINAL, NOMINAL]
#     for code in OCC_CODES:
#         for attr in DESIRED_ATTRS[2:]:
#             if attr in NOMINAL_ATTRS:
#                 column_types.append(NOMINAL)
#             else:
#                 column_types.append(CONTINUOUS)
#     return column_types

def normalize_data(X):
    X = np.array(X, dtype=float)
    normalized_data = []
    for i in range(len(X[0])):
        data = X[:, i]
        x_min = np.nanmin(data)
        x_max = np.nanmax(data)
        max_minus_min = x_max - x_min
        normalized_column = (data - x_min) / max_minus_min
        normalized_data.append(normalized_column)
    return np.array(normalized_data).T

def load_data(csv_save_path):
    try:
        selected_data = pd.read_csv(csv_save_path)
    except:
        data = hd(OCC_CODES, "Metro_average_all.csv", "MSA_master_clean.csv", DESIRED_ATTRS)
        selected_data = data.data
        selected_data.to_csv(csv_save_path, header=selected_data.columns, index=None)
    return selected_data

def one_hot_encode_states(dataset_frame):
    state_columns = [x for x in dataset_frame.columns if STATE_ABREVS in x]
    if len(state_columns) == 0:
        return dataset_frame
    state_abrevs_column = dataset_frame[state_columns[0]]
    encoded_states = pd.get_dummies(state_abrevs_column, prefix=STATE_ABREVS) # oneHotEncode states
    reformated_dataframe = dataset_frame.drop(state_columns, axis=1) # remove string states
    reformated_dataframe = reformated_dataframe.join(encoded_states) # append oneHotEncoded states
    housing_data = reformated_dataframe[HOUSING_DATA]
    reformated_dataframe = reformated_dataframe.drop(HOUSING_DATA, axis=1)
    reformated_dataframe = reformated_dataframe.join(housing_data) # add housingData back to the last position
    return reformated_dataframe

def normalize_and_clean(input_attributes):
    normed_input = normalize_data(input_attributes)
    normed_input[np.isnan(np.array(normed_input, dtype=np.float64))] = 1
    return normed_input

def sklearn_knn(dataset_frame):
    dataset = np.array(dataset_frame)
    data = normalize_and_clean(dataset[:, 2:-1])
    labels = dataset[:, -1]
    data, test_data, labels, test_labels = train_test_split(data, labels, train_size=.75, shuffle=True)
    predictions = KNNRegressor(n_neighbors=3).fit(data, labels).predict(test_data)
    accuracy = MSE(test_labels, predictions) ** (1/2)
    print("Accuracy = [{:.4f}]".format(accuracy))

# This modifies the data so that the differential makes the Salary data "year_differential" years OLDER than the housing data.
# If you want it in reverse, feed in a negative number.
def divide_and_regroup_year_differential(dataset_frame, year_differential=0):
    dataset = one_hot_encode_states(dataset_frame)
    housing_data = dataset[['AREA_NAME', 'YEAR', 'HOUSING_INDEX']]
    salary_data = dataset.drop(['HOUSING_INDEX'], axis=1)
    salary_data['YEAR'] = salary_data['YEAR'] + year_differential
    return salary_data.merge(housing_data)

def main():
    dataset = load_data("CS_job_correlation_select_values.csv")
    for i in range(7):
        modified_dataset = divide_and_regroup_year_differential(dataset, i)
        print("Year Differential: {:d}".format(i))
        sklearn_knn(modified_dataset)
    x = 1

if __name__=="__main__":
    main()
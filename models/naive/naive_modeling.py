import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from data.loader import HousingDataset as hd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse



# data is a hd object
def get_baseline_acc_2017(data):
    rmse_list = []
    diff_list = []
    for name, df in data.iterate_areas():
        if 2017 in df['YEAR']:
            # Index in 2017
            x = df.loc[(df['YEAR'] == 2017)]['HOUSING_INDEX']

            # Index in 2018
            y = df.loc[(df['YEAR'] == 2018)]['HOUSING_INDEX']

            rmse = np.sqrt(mse(y, x))
            diff = float(x) / float(y)

            rmse_list.append(rmse)
            diff_list.append(diff)

    print('Baseline 2017 -> 2018:', np.mean(rmse_list), np.mean(diff_list))


def get_baseline_acc_all(data):
    rmse_list = []
    diff_list = []
    indexes = []
    time_differences = []
    for name, df in data.iterate_areas():
        for year in range(2004, 2018):
            if year in list(df['YEAR']) and year+1 in list(df['YEAR']):
                x = df.loc[(df['YEAR'] == year)]['HOUSING_INDEX'].item()
                y = df.loc[(df['YEAR'] == year + 1)]['HOUSING_INDEX'].item()
                # import pdb; pdb.set_trace()
                rmse = y - x
                diff = float(x) / float(y)

                rmse_list.append(rmse)
                diff_list.append(diff)

                indexes.append(float(x))
                time_differences.append(float(y)-float(x))

    print('Baseline all years:', np.sqrt(np.nanmean(np.square(rmse_list))), np.mean(diff_list))
    return rmse_list

    # plt.subplot(121)
    # plt.plot(range(2004, 2018), indexes[:14])
    # plt.title('Original data')
    # plt.xlabel('Year')

    # plt.subplot(122)
    # plt.plot(range(2004, 2018), time_differences[:14])
    # plt.title('Time-differenced data')
    # plt.xlabel('Year')
    # plt.ylim(-20000, 20000)
    # plt.show()
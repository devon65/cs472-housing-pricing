import pandas as pd
import numpy as np
START_YEAR = 2002
END_YEAR = 2021
MONTH_STRINGS = ["-01", "-02", "-03", "-04", "-05", "-06", "-07", "-08", "-09", "-10", "-11", "-12"]


def extract_yearly_averages(housing_data):
    yearly_averages = np.matrix(housing_data['State']).T
    years = []
    for year in range(START_YEAR, END_YEAR):
        monthly_pricing = []
        years.append(str(year))
        for month in MONTH_STRINGS:
            month_str = str(year) + month
            if month_str in housing_data:
                monthly_pricing.append(housing_data[month_str])
        monthly_pricing = np.array(monthly_pricing)
        year_average = np.nanmean(monthly_pricing, axis=0)
        year_average = np.matrix(year_average).T
        # year_average = year_average.T
        yearly_averages = np.concatenate((yearly_averages, year_average), axis=1)
    return np.array(yearly_averages), years


def replace_avg_months_with_years(housing_data, yearly_averages):
    yearly_housing_data = np.array(housing_data['State'])
    for year in yearly_averages:
        yearly_housing_data = np.append(yearly_housing_data, yearly_averages[year], axis=0)
    return yearly_housing_data


def average_across_states(yearly_housing_data):
    states = np.unique(yearly_housing_data[:, 0])
    yearly_avg_by_state = []
    for state in states:
        state_data = [datapoint for datapoint in yearly_housing_data if datapoint[0] == state]
        state_data = np.array(state_data)[:, 1:]
        state_yearly_data = [state]
        state_averages = np.nanmean(state_data, dtype=float, axis=0).astype(int)
        state_yearly_data = np.append(state_yearly_data, state_averages)
        yearly_avg_by_state.append(state_yearly_data)
    return np.array(yearly_avg_by_state)

def write_matrix_to_csv(matrix, headers, file_name):
    pd.DataFrame(matrix).to_csv(file_name, header=headers, index=None)

def main():
    housing_data = pd.read_csv("Zip_Zhvi_AllHomes.csv", encoding='latin-1')
    yearly_averages, years = extract_yearly_averages(housing_data)
    yearly_avg_by_state = average_across_states(yearly_averages)
    headers = np.append(['State'], years)
    write_matrix_to_csv(yearly_avg_by_state, headers, 'yearly_housing_by_state1.csv')
    yas=1

if __name__ == '__main__':
    main()
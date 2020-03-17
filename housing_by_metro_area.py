import pandas as pd
import numpy as np

CITY_KEY = "City"
STATE_KEY = "State"
METRO_AREA_KEY = "Metro"
COUNTY_KEY = "CountyName"
START_YEAR = 1997
END_YEAR = 2021
MONTH_STRINGS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]


def stack_data_by_date(housing_dataframe, area_key=METRO_AREA_KEY, make_files_by_year=True, make_files_by_month=False): #area_key is the key by which to average data per month
    headers = ("AREA_NAME", "YEAR", "MONTH", "HOUSING_INDEX")
    area_list = combine_area_and_state_names(housing_dataframe, area_key)
    master_housing_data = []
    for year in range(START_YEAR, END_YEAR):
        year_strings = np.full((len(area_list), 1), year)
        yearly_housing_data = []
        for month in MONTH_STRINGS:
            month_str = str(year) + '-' + month
            if month_str in housing_dataframe:
                month_strings = np.full((len(area_list), 1), month)
                month_values = np.matrix(housing_dataframe[month_str]).T
                month_housing_data = np.array(np.concatenate((area_list, year_strings, month_strings, month_values), 1))
                monthly_housing_data = average_across_column_index(month_housing_data, 0, 3)
                yearly_housing_data.extend(monthly_housing_data)
                if make_files_by_month:
                    write_matrix_to_csv(monthly_housing_data, headers, area_key + "_average_" + month_str + ".csv")
        master_housing_data.extend(yearly_housing_data)
        if make_files_by_year:
            write_matrix_to_csv(yearly_housing_data, headers, area_key + "_average_" + str(year) + ".csv")
    write_matrix_to_csv(master_housing_data, headers, area_key + "_average_all.csv")
    return master_housing_data

def combine_area_and_state_names(housing_dataframe, area_key):
    area_list = housing_dataframe[area_key]
    if area_key == STATE_KEY:
        return np.array(np.matrix(area_list).T)

    combined_names = []
    state_list = housing_dataframe[STATE_KEY]
    for area, state in zip(area_list, state_list):
        combined_names.append([str(area) + ', ' + state + ' MSA'])

    return np.array(combined_names)


def average_across_column_index(housing_data, unique_areas_index, values_to_average_index):
    housing_data = np.array(housing_data)
    area_data = housing_data[:, unique_areas_index]
    area_data = [x for x in area_data if str(x) != 'nan']
    unique_areas = np.unique(area_data)
    averages_for_areas = []
    for area in unique_areas:
        area_datapoints = [datapoint for datapoint in housing_data if datapoint[unique_areas_index] == area]
        vals_to_average = np.array(area_datapoints)[:, values_to_average_index].astype(float)
        average = np.nanmean(vals_to_average, dtype=float, axis=0).astype(int)
        area_average_datapoint = area_datapoints[0]
        area_average_datapoint[values_to_average_index] = average
        if average > 0:
            averages_for_areas.append(area_average_datapoint.tolist())
    return averages_for_areas


def write_matrix_to_csv(matrix, headers, file_name):
    pd.DataFrame(matrix).to_csv(file_name, header=headers, index=None)


def main():
    housing_dataframe = pd.read_csv("Zip_Zhvi_AllHomes.csv", encoding='latin-1')
    housing_data_stacked = stack_data_by_date(housing_dataframe)
    yas=1

if __name__ == '__main__':
    main()
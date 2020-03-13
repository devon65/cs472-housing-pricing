import numpy as np
import pandas as pd
import os
from glob import glob

'''
Columns dropped in later files:
 - JOBS 1000
 - Location Quartile
 '''

# Until 2014, the data was stored in 3 separate csvs
def get_file_names():
    file_names = glob('BLS_raw_data/*/MSA*')

    return sorted(file_names)


def make_dataframes():
    file_names = get_file_names()

    master_df = None

    first_cols = None
    # Years where 3 files need to be combined
    curr = None
    for i in range(len(file_names) - 5):
        print('\ni', i)
        df = pd.read_excel(file_names[i])
        if first_cols is None:
            first_cols = df.columns
        df.rename(columns={'OCC_GROUP': 'GROUP'}, inplace=True)
        diff = list(set(df.columns) - set(first_cols))
        df.drop(diff, axis=1, inplace=True)

        year = str(2004 + i//3)
        df.insert(0, 'YEAR', year)

        # Combine the 3 csvs into 1
        if curr is None:
            curr = df
        else:
            curr = curr.append(df, ignore_index=True)

        if i % 3 == 2:
            path = 'BLS_clean_data/MSA_' + year + '_clean.csv'
            curr.to_csv(path, index=False)

            if master_df is None:
                master_df = curr
            else:
                master_df = master_df.append(curr, ignore_index=True)

            curr = None


    
    # Years with just one file
    for i in range(len(file_names)-5, len(file_names)):
        print('\ni', i)
        df = pd.read_excel(file_names[i])
        if first_cols is None:
            first_cols = df.columns
            master_df = df
        df.rename(columns={'OCC_GROUP': 'GROUP'}, inplace=True)
        diff = list(set(df.columns) - set(first_cols))

        df.drop(diff, axis=1, inplace=True)
        year = str(2014 + (i-(len(file_names)-5)))
        df.insert(0, 'YEAR', year)

        path = 'BLS_clean_data/MSA_' + year + '_clean.csv'
        df.to_csv(path, index=False)

        master_df = master_df.append(df, ignore_index=True)

    master_df.to_csv('BLS_clean_data/MSA_master_clean.csv', index=False)
        


if __name__=="__main__":
    make_dataframes()
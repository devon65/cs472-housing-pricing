import pandas as pd
import numpy as np
from enum import Enum
import typing as t

class Feature(Enum):
    AREA = "AREA_NAME"
    YEAR = "YEAR"
    MONTH = "MONTH"
    HOUSING_INDEX = "HOUSING_INDEX"

class HousingDataset():
    zillow = None
    bls = None

    def __init__(self, occ_codes:t.List[str]):
        # 1. Ensure valid options 
        # 2. Ensure that CSV files exist else download
        # 3. Load and organize requested data
        if HousingDataset.zillow is None:
            zillow = pd.read_csv("data/Metro_average_all.csv")
            # zillow["AREA_NAME"] = zillow["AREA_NAME"].str[:-4]
            zillow["AREA_NAME"] = zillow["AREA_NAME"].str.strip()
            zillow["AREA_NAME"] = zillow["AREA_NAME"].str[:-4].where(zillow["AREA_NAME"].str.endswith(" MSA"), zillow["AREA_NAME"])
            HousingDataset.zillow = zillow
        else:
            zillow = HousingDataset.zillow

        self.zillow = zillow

        zillow_yearly_avg = zillow.groupby(["AREA_NAME", "YEAR"]).mean()
        zillow_yearly_avg.drop("MONTH", inplace=True, axis=1)

        self.yearly_zillow = zillow_yearly_avg.reset_index().set_index(["AREA_NAME", "YEAR"])

        # Remove MSA
        if HousingDataset.bls is None:
            bls_data = pd.read_csv("data/MSA_master_clean.csv").reset_index()
            numeric_cols = ['YEAR', 'TOT_EMP', 'EMP_PRSE', 'H_MEAN', 'A_MEAN', 'MEAN_PRSE', 'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90', 'ANNUAL', 'HOURLY']
            
            for col in numeric_cols:
                bls_data[col] = pd.to_numeric(bls_data[col], errors="coerce")

            bls_data["AREA_NAME"] = bls_data["AREA_NAME"].str.strip()
            bls_data["AREA_NAME"] = bls_data["AREA_NAME"].str[:-4].where(bls_data["AREA_NAME"].str.endswith(" MSA"), bls_data["AREA_NAME"])
            HousingDataset.bls = bls_data
        else:
            bls_data = HousingDataset.bls

        self.bls_data = bls_data

        desired_bls_cols = \
        ['YEAR', 'PRIM_STATE', 'AREA', 'AREA_NAME', 'TOT_EMP', 'EMP_PRSE', 'H_MEAN', 'A_MEAN', 'MEAN_PRSE', 'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90', 'ANNUAL', 'HOURLY']

        join_builder = None

        for occ in occ_codes:
            print(f"Adding {occ}")
            subset = bls_data.loc[bls_data["OCC_CODE"] == occ]
            subset = subset[desired_bls_cols].set_index(["AREA_NAME", "YEAR"])
            if join_builder is None:
                join_builder = subset
            else:
                join_builder = join_builder.join(subset, how="outer", rsuffix=f"_{occ}")
        
        self.bls = join_builder

        self.data = self.bls.join(self.yearly_zillow, how="inner").reset_index()
            
    def __getitem__(self, i):
        return self.data.loc[i]

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self._current_next is None:
            self._current_next = -1
        
        if self._current_next >= len(self):
            raise StopIteration
        else:
            self._current_next += 1
            return self[self._current_next]

    def iterate_as_numpy(self):
        for x in self:
            yield np.array(x)

    def iterate_areas(self):
        for x in self.data.groupby("AREA_NAME"):
            yield (x[0], x[1].sort_values("YEAR"))

    def columns(self):
        return self.data.columns
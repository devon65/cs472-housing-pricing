import pandas as pd
import numpy as np
from enum import Enum
import typing as t

from data.datasets.zillow import Zillow
from data.datasets.bls import BLS

class Feature(Enum):
    AREA = "AREA_NAME"
    YEAR = "YEAR"
    MONTH = "MONTH"
    HOUSING_INDEX = "HOUSING_INDEX"


class HousingDataset():
    zillow = None
    bls = None

    def __init__(self, occ_codes: t.List[str]):
        # 1. Ensure valid options
        # 2. Ensure that CSV files exist else download
        # 3. Load and organize requested data
        zillow = Zillow().load()
        self.zillow = zillow

        zillow_yearly_avg = zillow.groupby(["AREA_NAME", "YEAR"]).mean()
        zillow_yearly_avg.drop("MONTH", inplace=True, axis=1)

        self.yearly_zillow = zillow_yearly_avg.reset_index().set_index(["AREA_NAME", "YEAR"])

        bls_data = BLS().load()
        self.bls_data = bls_data

        desired_bls_cols = \
            ['YEAR', 'PRIM_STATE', 'AREA', 'AREA_NAME', 'TOT_EMP', 'EMP_PRSE', 'H_MEAN', 'A_MEAN', 'MEAN_PRSE',
             'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75',
             'A_PCT90', 'ANNUAL', 'HOURLY']

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

    def iterate_areas_with_window(self, window_size:int=5):
        for city, data in self.iterate_areas():
            for window, years in HousingDataset._slide_window(data, window_size):
                yield (city, years, window)

    def iterate_areas_with_flat_window(self, window_size:int=5, make_target:bool=False):
        for city, (first_year, last_year), data in self.iterate_areas_with_window(window_size):
            # import pdb; pdb.set_trace()
            data = data.reset_index().set_index("AREA_NAME")
            join_builder = pd.DataFrame([city], columns=["AREA_NAME"]).set_index("AREA_NAME")

            for year in range(first_year, last_year + (1 if not make_target else 0)):
                subset = data.loc[data["YEAR"] == year]
                join_builder = join_builder.join(subset, how="outer", rsuffix=f"_{year - first_year}")

            if make_target:
                target = data.loc[data["YEAR"] == last_year]
                yield city, (first_year, last_year), join_builder, target
            else:
                yield city, (first_year, last_year), join_builder

    @staticmethod
    def _slide_window(data:pd.DataFrame, window_size:int):
        first_year = data["YEAR"].min()
        last_year = data["YEAR"].max()
        year_series = pd.DataFrame(list(range(first_year, last_year + 1)), columns=["YEAR"]).set_index("YEAR")
        data_with_consecutive_years = data.set_index("YEAR").join(year_series, how="outer")

        for i in range(len(data_with_consecutive_years) - window_size):
            begin_year = first_year + i
            last_year = begin_year + window_size - 1
            yield ( data_with_consecutive_years[i:i+window_size], (begin_year, last_year) )


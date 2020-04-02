import pandas as pd

class Zillow():
    zillow = None

    def load(self):
        if Zillow.zillow is None:
            zillow = pd.read_csv("data/Metro_average_all.csv")
            # zillow["AREA_NAME"] = zillow["AREA_NAME"].str[:-4]
            zillow["AREA_NAME"] = zillow["AREA_NAME"].str.strip()
            zillow["AREA_NAME"] = zillow["AREA_NAME"].str[:-4].where(zillow["AREA_NAME"].str.endswith(" MSA"), zillow["AREA_NAME"])
            Zillow.zillow = zillow

        return Zillow.zillow
import pandas as pd

class BLS():
    bls = None
    def load(self):
        # Remove MSA
        if BLS.bls is None:
            bls_data = pd.read_csv("data/MSA_master_clean.csv").reset_index()
            numeric_cols = ['YEAR', 'TOT_EMP', 'EMP_PRSE', 'H_MEAN', 'A_MEAN', 'MEAN_PRSE', 'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 'H_PCT90', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90', 'ANNUAL', 'HOURLY']
            
            for col in numeric_cols:
                bls_data[col] = pd.to_numeric(bls_data[col], errors="coerce")

            bls_data["AREA_NAME"] = bls_data["AREA_NAME"].str.strip()
            bls_data["AREA_NAME"] = bls_data["AREA_NAME"].str[:-4].where(bls_data["AREA_NAME"].str.endswith(" MSA"), bls_data["AREA_NAME"])
            BLS.bls = bls_data

        return BLS.bls
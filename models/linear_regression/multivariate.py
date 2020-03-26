import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from data.loader import HousingDataset as hd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from itertools import product

def regress_on_data(data:pd.DataFrame, independent_vars:[str], lag:int):
    """
        Returns the difference between the prediction of the last
        element and the known value.
    """
    x = data[independent_vars].to_numpy()
    if lag > 0:
        x = x[:-lag]
    y = data["HOUSING_INDEX"].to_numpy()[lag:]
    known_x = x[:-1]
    known_y = y[:-1]
    unknown_x = x[-1].reshape(1, -1)
    unknown_y = y[-1].item()

    poly = PolynomialFeatures(degree=len(independent_vars))
    known_x_t = poly.fit_transform(known_x)
    unknown_x_t = poly.fit_transform(unknown_x)

    clf = linear_model.LinearRegression()
    pred = clf.fit(known_x_t, known_y).predict(unknown_x_t).item()

    return pred - unknown_y

def regress_on_all(occ_code:str, independent_columns:[str], lag:int):
    data = hd([occ_code])
    data.data.dropna(subset=independent_columns, inplace=True)

    ind = []
    for i, city in enumerate(data.iterate_areas()):
        if len(city[1]) < 5:
            continue
        ind.append(regress_on_data(city[1], independent_columns, lag))

    squares = np.square(np.array(ind))
    mean = np.mean(squares)
    rmse = np.sqrt(mean)
    print(len(ind))
    return rmse, ind

def find_best_combo():
    col_combos = [
        ["YEAR"],
        ["YEAR", "TOT_EMP"],
        ["TOT_EMP"],
        ["YEAR", "H_MEDIAN"],
        ["YEAR", "A_MEDIAN"],
        ["YEAR", "H_PCT10"],
        ["TOT_EMP", "H_MEDIAN"]
    ]
    occ_codes = [ f"{i:02d}-0000" for i in range(0, 55)]
    lags = [0, 1, 2]
    prod = list(product(occ_codes, col_combos, lags)) + [( "00-0000", ["HOUSING_INDEX"], 1)] + [( "00-0000", ["HOUSING_INDEX"], 2)]

    results, actuals = zip(*[ regress_on_all(occ, combo, lag) for occ, combo, lag in prod ])

    with open("/tmp/all_results.csv", "w") as f:
        for (o, c, l), r in zip(prod, results):
            print(f"{o}, {c} {l} {r}")
            f.write(f"{o}\t{c}\t{l}\t{r}\n")

    print(f"Best degree: {prod[np.nanargmin(results)]} with RMSE {min(results)}")

    actuals = np.array(actuals)

    plt.boxplot(actuals.tolist())
    plt.xticks(range(1, len(results) + 1), [ f"{o} {c} Lag: {l} : RMSE: {r:.2f}" for r, (o, c, l) in zip(results, prod)], rotation=90)
    plt.title("Error Variance Tests")
    plt.xlabel("Test")
    plt.ylabel("RMSE")
    plt.ylim(500000, -500000)
    plt.show(block=False)

    plt.bar(range(1, len(results) + 1), results)
    plt.xticks(range(1, len(results) + 1), [ f"{o} {c} Lag: {l} : RMSE: {r:.2f}" for r, (o, c, l) in zip(results, prod)], rotation=90)
    plt.title("RMSE Tests")
    plt.xlabel("Test")
    plt.ylabel("RMSE")
    plt.show()

    return results, actuals




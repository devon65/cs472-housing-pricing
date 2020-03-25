import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from data.loader import HousingDataset as hd
import matplotlib.pyplot as plt

def regress_on_data(data:pd.DataFrame, degree:int):
    """
        Returns the difference between the prediction of the last
        element and the known value.
    """
    # print(data)
    known_x = data["YEAR"].to_numpy()[:-1]
    known_y = data["HOUSING_INDEX"].to_numpy()[:-1]
    unknown_x = data["YEAR"].to_numpy()[-1].item()
    unknown_y = data["HOUSING_INDEX"].to_numpy()[-1].item()

    coefficients = polyfit(known_x, known_y, degree)

    # print([c for c in coefficients])
    z = zip(range(len(coefficients)), [c for c in coefficients])
    prediction = np.sum([ c * unknown_x**b  for b, c in z])

    return unknown_y - prediction

def regress_on_all(degree:int):
    data = hd(["15-0000"])

    ind = []
    for i, city in enumerate(data.iterate_areas()):
        if len(city[1]) < 5:
            continue

        ind.append(regress_on_data(city[1], degree))

    squares = np.square(np.array(ind))
    mean = np.mean(squares)
    rmse = np.sqrt(mean)
    return rmse, ind

def find_best_degree():
    results, actuals = zip(*[ regress_on_all(d) for d in range(1, 6) ])

    print(f"Best degree: {np.argmin(results) + 1} with RMSE {min(results)}")

    plt.boxplot(np.abs(actuals).tolist())
    plt.plot(range(1, 6), results)
    # plt.xticks(range(0, 5), range(1, 6))
    plt.title("RMSE By Degree of Regression for Housing Index Only")
    plt.xlabel("Degree")
    plt.ylabel("RMSE")
    plt.show()




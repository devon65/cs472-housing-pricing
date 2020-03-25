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

    total = 0
    count = 0

    for i, city in enumerate(data.iterate_areas()):
        if len(city[1]) < 5:
            continue

        total += regress_on_data(city[1], degree)**2
        count += 1
        if i % 25 == 0:
            rmse = (total / count)**0.5
            print(f"RMSE so far: {rmse}")
    
    rmse = (total / count)**0.5
    print(f"RMSE: {rmse}")
    return rmse

def find_best_degree():
    results = [ regress_on_all(d) for d in range(1, 6) ]

    print(f"Best degree: {np.argmin(results) + 1} with RMSE {min(results)}")

    plt.plot(results)
    plt.xticks(range(0, 5), range(1, 6))
    plt.title("RMSE By Degree of Regression for Housing Index Only")
    plt.xlabel("Degree")
    plt.ylabel("RMSE")
    plt.show()



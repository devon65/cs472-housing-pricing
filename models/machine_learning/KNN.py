import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator, ClassifierMixin

NO_WEIGHT = 'no_weight'
INVERSE_DISTANCE = 'inverse_distance'
CONTINUOUS = 'continuous'
NOMINAL = 'nominal'

class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, column_type, k_mins=3, weight_type=INVERSE_DISTANCE, use_regression=False): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.column_type = column_type
        self.weight_type = weight_type
        self.k_mins = k_mins
        self.use_regression = use_regression

    def fit(self, X, y, normalize=True):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        if normalize:
            X = self.normalize_data(X)
        self.training_X = X
        self.training_y = np.array(y)[:,0]
        return self

    def normalize_data(self, X):
        X = np.array(X)
        if len(X[0]) != len(self.column_type):
            raise Exception('Given column_type and data have different number of columns')
        normalized_data = []
        for i in range(len(self.column_type)):
            if self.column_type[i] == NOMINAL: continue
            data = X[:, i]
            x_min = np.nanmin(data)
            x_max = np.nanmax(data)
            max_minus_min = x_max - x_min
            normalized_column = (data - x_min) / max_minus_min
            normalized_data.append(normalized_column)
        return np.array(normalized_data).T

    def predict_point(self, point):
        all_parts_distances = []
        for i in range(len(point)):
            if self.column_type[i] == CONTINUOUS:
                trainings = self.training_X[:, i]
                partial_distance = np.absolute(point[i] - trainings)
                partial_distance[np.isnan(np.array(partial_distance, dtype=np.float64))] = 1
            else:
                partial_distance = [0 if t == point[i] else 1 for t in self.training_X[:, i]]
            all_parts_distances.append(partial_distance)

        squared_dists = np.square(all_parts_distances)
        summed_squared_dists = np.sum(squared_dists, 0)
        final_distances = np.sqrt(np.array(summed_squared_dists, dtype=np.float64))
        mindices = np.argpartition(final_distances, self.k_mins)[:self.k_mins]

        if self.weight_type == INVERSE_DISTANCE:
            return self.inverse_distance_weighted_voting(summed_squared_dists, mindices)
        else:
            return self.no_weight_voting(mindices)

    def inverse_distance_weighted_voting(self, squared_distances, mindices):
        inverse_distances = 1/squared_distances
        min_targets = self.training_y[mindices]
        minverse_distances = inverse_distances[mindices]
        if self.use_regression:
            weighted_targets = min_targets * minverse_distances
            return (np.sum(weighted_targets))/np.sum(minverse_distances)
        else:
            highest_vote = 0
            prediction = min_targets[0]
            for unique_target in np.unique(min_targets):
                dists_indices = np.where(min_targets == unique_target)[0]
                contending_vote = np.sum(minverse_distances[dists_indices])
                if contending_vote > highest_vote:
                    highest_vote = contending_vote
                    prediction = unique_target
            return prediction

    def no_weight_voting(self, mindices):
        min_targets = self.training_y[mindices]
        if self.use_regression:
            return np.average(min_targets)
        else:
            return stats.mode(min_targets).mode[0]

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        return [self.predict_point(point) for point in X]

    def score(self, X, y, normalize=True):
        # Returns the Mean score given input data and labels
        """ Return accuracy of model on a given dataset. Must implement own score function.
            Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets
            Returns:
                score : float
                        Mean accuracy of self.predict(X) wrt. y.
                        :param **kwargs:
        """
        if normalize:
            X = self.normalize_data(X)
        predicted = self.predict(X)
        if self.use_regression:
            return self.find_mse(predicted, y), predicted
        else:
            return self.compare_predictions(predicted, y), predicted

    @staticmethod
    def compare_predictions(predicted, actual):
        number_correct = 0
        for pred, act in zip(predicted, actual):
            if pred == act[0]:
                number_correct = number_correct + 1
        return number_correct / len(actual)

    @staticmethod
    def find_mse(predictions, actual_targets):
        squared = (actual_targets[:, 0] - predictions) ** 2
        return np.average(np.array(squared, dtype=np.float64))
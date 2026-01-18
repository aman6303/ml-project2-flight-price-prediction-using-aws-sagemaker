# creating a class for rbf percentile similarity

import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

sklearn.set_config(transform_output="pandas")


class RBFPercentileSimilarity(
    BaseEstimator, TransformerMixin
):  # we inherit from these two to make it sklearn compatible
    def __init__(
        self, variables=None, percentiles=[0.25, 0.5, 0.75], gamma=0.1
    ):  # gamma is to control underfitting and overfittung
        self.variables = variables
        self.percentiles = percentiles
        self.gamma = gamma

    def fit(self, X, y=None):
        if not self.variables:
            self.variables = X.select_dtypes(include="number").columns.to_list()

        self.reference_values_ = {
            col: (X.loc[:, col].quantile(self.percentiles).values.reshape(-1, 1))
            for col in self.variables
        }

        return self  # to make it sklearn compatible

    # Returning self allows transform() to reuse them downstream without recomputing.

    def transform(self, X):
        objects = []
        for col in self.variables:
            columns = [
                f"{col}_rbf_{int(percentile*100)}" for percentile in self.percentiles
            ]
            obj = pd.DataFrame(
                data=rbf_kernel(
                    X.loc[:, [col]], Y=self.reference_values_[col], gamma=self.gamma
                ),  # calculating rbf kernal based on reference value
                columns=columns,
            )
            objects.append(obj)
        return pd.concat(objects, axis=1)

"""
Class for the computation of the Aggregated demand curve.
The aggregated curve is composed by the weighted sum of the classes curves
"""

import numpy as np


class AggregatedCurve:

    def __init__(self, curve_values, weights):
        """
        Initialization of the aggregate curve
        :param curve_values: the classes curves
        :param weights: the weights for the weighted sum
        """
        self.curve_values = curve_values
        self.weights = weights
        self.agg = self.create_aggregate_demand_curve()

    # TODO: creare un metodo che gestisca un numero di classi variabile
    def create_aggregate_demand_curve(self):
        """
        Creation of the aggregated demand curve
        :return: a list with all phases aggregated curves
        """
        agg_curve_values = [[[], []], [[], []], [[], []], [[], []]]
        # for readability
        f = self.curve_values[0]
        s = self.curve_values[1]
        t = self.curve_values[2]
        for p in range(len(f)):  # for all the phases
            for j in range(len(f[p])):  # for all the axis
                for i in range(len(f[p][j])):  # for all the element of the x and y
                    agg_curve_values[p][j].append(
                        np.average(a=np.array([f[p][j][i], s[p][j][i], t[p][j][i]]), weights=self.weights))
        return agg_curve_values

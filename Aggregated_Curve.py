import numpy as np


class Aggregated_Curve():

    def __init__(self, curve_values, weights):
        self.curve_values = curve_values
        self.weights = weights
        self.agg = self.create_aggregate_demand_curve()


    #TODO: creare un metodo che gestisca un numero di classi variabile
    def create_aggregate_demand_curve(self):
        agg_curve_values = [[[], []], [[], []], [[], []], [[], []]]
        #for readability
        f = self.curve_values[0]
        s = self.curve_values[1]
        t = self.curve_values[2]
        for p in range(len(f)): # for all the phases
            for j in range(len(f[p])): #for all the axis
                for i in range(len(f[p][j])): # for all the element of the x and y
                    agg_curve_values[p][j].append(
                        np.average(a=np.array([f[p][j][i], s[p][j][i], t[p][j][i]]), weights=self.weights))
        return agg_curve_values


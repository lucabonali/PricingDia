import numpy as np
from DemandCurve import *
import Data


'''
Control function
'''
def checkLen(f, s, t):
    if (len(f[0]) == len(s[0]) == len(t[0])):
        if (len(f[1]) == len(s[1]) == len(t[1])):
            return True
    return False


'''
Computation of the aggregated curve
- f: first curve
- s: second curve
- t: third curve
- weights: probabilities of each class
'''
def create_aggregate_demand_curve(f, s, t, weights):
    agg_curve_values = [[[], []], [[], []], [[], []], [[], []]]
    for k in range(len(f)):
        for j in range(len(f[k])):
            for i in range(len(f[k][j])):
                agg_curve_values[k][j].append(
                    np.average(a=np.array([f[k][j][i], s[k][j][i], t[k][j][i]]), weights=weights))
    return agg_curve_values


'''
Demand curve creations:
one for each class and the aggregated one
'''
first_curve = Data.first_curve_values
second_curve = Data.second_curve_values
third_curve = Data.third_curve_values

agg_curve = create_aggregate_demand_curve(Data.first_curve_values, Data.second_curve_values,
                                                      Data.third_curve_values, Data.weights)
curves = [first_curve, second_curve, third_curve, agg_curve]


'''
Plot the demand curves of the selected phase
phase: the selected phase
'''
def plot_phase_curves(phase):
    plt.figure(0)
    plt.xlabel("Price")
    plt.ylabel("Demand")
    plt.title('Demand curves of phase: %i' %phase)
    for i in range(len(curves)):
        plt.plot(curves[i][phase][1], curves[i][phase][0], linewidth=1, markersize=1)
    plt.legend(["Class 1", "Class 2", "Class 3", "Aggregated"])
    plt.show()


'''
Plot of the demand curves for all the phases
'''
for p in range(len(curves[0])):
    plot_phase_curves(p)

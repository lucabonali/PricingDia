"""
Module with the data of the project:
    - Phases and time horizon
    - Demand curves
    - Margins
"""

from AggregatedCurve import *


'''
Phases and time horizon:
    - Launch phase: [Feb, Mar, Apr]
    - New competitor's product: [Set, Ott, Nov]
    - Holiday: [Dic, Gen]
    - New model: [Feb, Mar, Apr]
'''
# months_per_phases = [15, 0, 0, 0]
months_per_phases = [7, 3, 2, 3]
day_per_months = 30

samples_per_day = 20
samples_per_week = samples_per_day * 7
samples_per_month = samples_per_week * 4

day_per_phases = np.dot(months_per_phases, day_per_months)
samples_per_phase = np.dot(day_per_phases, samples_per_day)

samples_per_phase = [int(i) for i in samples_per_phase]

# Time Horizon
n_months = sum(months_per_phases)
t_horizon = n_months * day_per_months * samples_per_day

'''
Demand curves values of the form:
[[phase 1 prob., prices], [phase 2 prob., prices]...]    
'''
# Prices equal for all the phases
x_values = [0, 100, 200, 300, 400, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000, 1100, 1200, 1300, 1400]
n_candidates = len(x_values)

# Class 1 curve values
first_curve_values = [
    [[1, 0.95, 0.9, 0.9, 0.8, 0.7, 0.62, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.57, 0.5, 0.1, 0.05, 0.05, 0.05], x_values],
    [[0.9, 0.87, 0.82, 0.8, 0.7, 0.65, 0.55, 0.5, 0.45, 0.32, 0.27, 0.2, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01],
     x_values],
    [[0.9, 0.87, 0.82, 0.8, 0.7, 0.65, 0.55, 0.55, 0.5, 0.45, 0.35, 0.3, 0.25, 0.15, 0.1, 0.05, 0.01, 0.01, 0.01],
     x_values],
    [[0.9, 0.87, 0.87, 0.85, 0.5, 0.3, 0.25, 0.2, 0.18, 0.03, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0], x_values]]

# Class 2 curve values
second_curve_values = [
    [[1, 0.95, 0.93, 0.9, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.025, 0.01, 0.01, 0, 0, 0], x_values],
    [[0.95, 0.8, 0.65, 0.6, 0.35, 0.17, 0.15, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0], x_values],
    [[0.95, 0.82, 0.75, 0.72, 0.4, 0.17, 0.15, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0], x_values],
    [[0.95, 0.9, 0.85, 0.75, 0.4, 0.05, 0.05, 0.04, 0.04, 0.02, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0], x_values]]

# Class 3 curve values
third_curve_values = [
    [[0.55, 0.52, 0.5, 0.35, 0.3, 0.29, 0.28, 0.27, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.05, 0.07, 0.1, 0.06, 0.02],
     x_values],
    [[0.45, 0.45, 0.4, 0.3, 0.26, 0.25, 0.23, 0.23, 0.2, 0.15, 0.1, 0.08, 0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02],
     x_values],
    [[0.55, 0.52, 0.5, 0.35, 0.3, 0.23, 0.18, 0.15, 0.13, 0.1, 0.08, 0.1, 0.07, 0.05, 0.05, 0.07, 0.1, 0.06, 0.02],
     x_values],
    [[0.47, 0.47, 0.42, 0.35, 0.28, 0.25, 0.23, 0.23, 0.2, 0.15, 0.1, 0.08, 0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02],
     x_values]]

classes_curve_values = [first_curve_values, second_curve_values, third_curve_values]

# Weights: probabilities to a user to belong to a certain class
weights = [0.375, 0.425, 0.2]

# Aggregated Curve
agg_curve = AggregatedCurve(classes_curve_values, weights).agg

# All curve values (with also the aggregate)
classes_curve_values.append(agg_curve)
total_curve_values = classes_curve_values


def get_all_probabilities():
    return np.array(total_curve_values).squeeze()


def get_class_probabilities(n_class):
    """
    Get the probabilities for all the phases of a specific class:
        - n_class = 0: return first class probs.
        - n_class = 1: return second class probs.
        - n_class = 2: return third class probs.
        - n_class = 3: return aggregated curve probs.
    :param n_class: the selected class
    :return: an array with the probabilities for all the phases
    """
    res = []

    for p in range(len(total_curve_values[n_class])):
        res.append(total_curve_values[n_class][p][0])

    return np.array(res)


'''
Margins: what we'll gain selling one unit of product at a certain price with a fixed production cost
'''



def minus_cost(values):
    """
    Margin computation
    :param values: prices list
    :return: margins list
    """
    return np.array(values) - cost_of_unit


# Cost of one unit of product
cost_of_unit = 350

margins = minus_cost(x_values)


'''
K-testing candidates
'''
k_testing_candidates = [[(x_values[i], agg_curve[p][0][i], margins[i]) for i in range(len(x_values))] for p in
                        range(len(first_curve_values))]


# mettere un parametro che si riferisce al numero di samples, in modo da provare un po' per quali valori è meglio
# l'aggragate e per qualiè meglio il disaggregate. Così da vedere quale usare nel caso in cui nel mondo realte
# ne possiamo ottenere tot al giorno bias variance tradeoff

from Aggregated_Curve import *

'''
Curves values
'''

x_values = [0, 100, 200, 300, 400, 500, 550, 600, 650, 700, 750, 800, 850, 900, 1000, 1100, 1200, 1300, 1400]

# curve for the youngsters
first_curve_values = [
    [[1, 0.95, 0.9, 0.9, 0.8, 0.7, 0.62, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.57, 0.5, 0.1, 0.05, 0.05, 0.05],
     x_values],
    # launch phase feb, mar, apr
    [[0.9, 0.87, 0.82, 0.8, 0.7, 0.65, 0.55, 0.5, 0.45, 0.32, 0.27, 0.2, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01],
     x_values],
    # new competitor's product set, ott, nov
    [[0.9, 0.87, 0.82, 0.8, 0.7, 0.65, 0.55, 0.55, 0.5, 0.45, 0.35, 0.3, 0.25, 0.15, 0.1, 0.05, 0.01, 0.01, 0.01],
     x_values],
    # Holiday dic, gen
    [[0.9, 0.87, 0.87, 0.85, 0.5, 0.3, 0.25, 0.2, 0.18, 0.03, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0],
     x_values]]  # new model feb mar apr

# Curve for the adults
second_curve_values = [
    [[1, 0.95, 0.93, 0.9, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.025, 0.01, 0.01, 0, 0, 0],
     x_values],
    [[0.95, 0.8, 0.65, 0.6, 0.35, 0.17, 0.15, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0],
     x_values],
    [[0.95, 0.82, 0.75, 0.72, 0.4, 0.17, 0.15, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0],
     x_values],
    [[0.95, 0.9, 0.85, 0.75, 0.4, 0.05, 0.05, 0.04, 0.04, 0.02, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0],
     x_values]]

# Curve for the third age
third_curve_values = [
    [[0.55, 0.52, 0.5, 0.35, 0.3, 0.29, 0.28, 0.27, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.05, 0.07, 0.1, 0.06, 0.02],
     x_values],
    [[0.45, 0.45, 0.4, 0.3, 0.26, 0.25, 0.23, 0.23, 0.2, 0.15, 0.1, 0.08, 0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02],
     x_values],
    [[0.55, 0.52, 0.5, 0.35, 0.3, 0.23, 0.18, 0.15, 0.13, 0.1, 0.08, 0.1, 0.07, 0.05, 0.05, 0.07, 0.1, 0.06, 0.02],
     x_values],
    [[0.47, 0.47, 0.42, 0.35, 0.28, 0.25, 0.23, 0.23, 0.2, 0.15, 0.1, 0.08, 0.05, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02],
     x_values]]

# To keep the demand curve dynamic
classes_curve_values = [first_curve_values, second_curve_values, third_curve_values]

# weights, as probability to belong to a certain class: Adults, Student or Old people
weights = [0.375, 0.425, 0.2]

# Aggregated Curve
agg_curve = Aggregated_Curve(classes_curve_values, weights).agg

# All curve values (with also the aggregate)
classes_curve_values.append(agg_curve)
total_curve_values = classes_curve_values

# cost of each unit
cost_of_unit = 350

# number of arms/candidates
n_arms = len(first_curve_values[0][1])


def minus_cost(values):
    return np.array(values) - cost_of_unit


'''
Margin: what we'll gain selling one unit of product at a certain price with a fixed production cost
In our case:

[-350 -250 -150  -50   50  150  200  250  300  350  400  450  500  550  650  750  850  950 1050]
'''
margins = minus_cost(x_values)


'''
K-testing stuff
'''
k_testing_candidates = [[(x_values[i], agg_curve[p][0][i], margins[i]) for i in range(len(x_values))] for p in range(len(first_curve_values))]

'''
Samples and phases
'''

# Samples per day
samples_per_day = 50

# phases
months_per_phases = [7, 3, 2, 3]
day_per_months = 30
day_per_phases = np.dot(months_per_phases, day_per_months)
samples_per_phase = np.dot(day_per_phases, samples_per_day)

# Time Horizon
n_months = sum(months_per_phases)
t_horizon = n_months * day_per_months * samples_per_day


# if n_class = 0 the method return the probability matrix of the first curve
# if n_class = 1 the method return the probability matrix of the second curve
# if n_class = 2 the method return the probability matrix of the third curve
# if n_class = 3 the method return the probability matrix of the aggregate curve
# the probability matrix have as rows the probabilities of the correspective phase
def get_class_probabilities(n_class):
    res = []
    for p in range(len(total_curve_values[n_class])):
        res.append(total_curve_values[n_class][p][0])
    return np.array(res)

# mettere un parametro che si riferisce al numero di samples, in modo da provare un po' per quali valori è meglio l'aggragate e
# per qualiè meglio il disaggregate. Così da vedere quale usare nel caso in cui nel mondo realte ne possiamo ottenere tot al giorno
# bias variance tradeoff

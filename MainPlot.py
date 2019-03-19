import numpy as np
from DemandCurve import *

first_curve_values = [[0.9,0.9,0.8,0.7,0.6,0.6,0.5,0.1,0.05,0.05],[200,300,400,500,600,700,1000,1100,1200,1300]] #curve for the youngsters
second_curve_values = [[0.93,0.9,0.7,0.5,0.3,0.15,0.1,0.05,0.025,0.01],[200,300,400,500,550,650,700,800,900,1000]] #Curve for the adults
third_curve_values = [[0.5,0.35,0.3,0.2,0.1,0.05,0.03,0.05,0.1,0.05],[200,300,400,700,800,850,1000,1200,1300,1400]] #Curve for the third age
weights = [0.375,0.425,0.2] # weights, as probability to belong to a certain class: Adults, Student or Old people

def checkLen(f,s,t):
    if(len(f[0]) == len(s[0]) == len(t[0])):
        if(len(f[1]) == len(s[1]) == len(t[1])):
            return True
    return False

def aggDemand(f,s,t,weights):
    agg_curve_values = [[],[]]
    for j in range(len(f)):
        for i in range(len(f[j])):
            agg_curve_values[j].append(np.average(a=np.array([f[j][i],s[j][i],t[j][i]]),weights=weights))
    return agg_curve_values

def plot_curves(f,s,t):
    first_curve = DemandCurve(first_curve_values)
    second_curve = DemandCurve(second_curve_values)
    third_curve = DemandCurve(third_curve_values)
    print(aggDemand(f, s, t, weights))
    agg_curve = DemandCurve(aggDemand(f, s, t, weights))

plot_curves(first_curve_values,second_curve_values,third_curve_values)
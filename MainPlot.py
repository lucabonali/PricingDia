import numpy as np

first_curve_values = [[],[]] #curve for the youngsters
second_curve_values = [[],[]] #Curve for the adults
third_curve_values = [[],[]] #Curve for the third age
weights = [0.375,0.425,0.2] # weights, as probability to belong to a certain class: Adults, Student or Old people

def checkLen(f,s,t):
    if(len(f[0]) == len(s[0]) == len(t[0])):
        if(len(f[1]) == len(s[1]) == len(t[1])):
            return True
    return False

def aggDemand(f,s,t,weights):
    agg_curve_values = [[],[]]
    for j in len(f):
        for i in len(f[j]):
            agg_curve_values.append(np.avg([f[j][i],s[j][i],t[j][i]],weights))
    return agg_curve_values



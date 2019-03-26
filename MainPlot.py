import matplotlib.pyplot as plt
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
Plot the demand curves of the selected phase
phase: the selected phase
'''
def plot_phase_curves(phase):
    plt.figure(0)
    plt.xlabel("Price")
    plt.ylabel("Demand")
    plt.title('Demand curves of phase: %i' %phase)
    for i in range(len(Data.total_curve_values)):
        plt.plot(Data.total_curve_values[i][phase][1], Data.total_curve_values[i][phase][0], linewidth=1, markersize=1)
    plt.legend(["Class 1", "Class 2", "Class 3", "Aggregated"])
    plt.show()


'''
Plot of the demand curves for all the phases
'''
for p in range(len(Data.total_curve_values[0])):
    plot_phase_curves(p)

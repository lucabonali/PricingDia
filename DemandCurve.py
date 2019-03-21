import matplotlib.pyplot as plt



'''
Ho lavorato direttamente con gli array di dati della classe Data
Ho lasciato il compito di disegnare a MainPlot
Lascio la classe nel caso ci venga in mente una soluzione più elegante
'''

class DemandCurve():
    def __init__(self, curve_values):
        self.curve_values = curve_values


    '''
    Return the values of the requested phase:
    - i: the requested phase
    '''
    def get_phase_values(self, i):
        return self.curve_values[i]


    '''
    Plot the demand curves of each phase
    '''
    def plot_phases_curve(self):
        plt.figure(0)
        plt.xlabel("Price")
        plt.ylabel("Demand")
        for i in range(len(self.curve_values)):
            plt.plot(self.curve_values[i][1], self.curve_values[i][0], linewidth=1, markersize=1)
        plt.legend(["phase 1", "phase 2", "phase 3", "phase 4"])
        plt.show()





    '''
    Plot the specific phase demand curve
    - i: the number of the phase
    '''
    def plot_single_phase_curve(self, i):
        plt.figure(0)
        plt.xlabel("Price")
        plt.ylabel("Demand")
        plt.plot(self.curve_values[i][1], self.curve_values[i][0], linewidth=1, markersize=1)
        plt.legend(["phase ",i])
        plt.show()
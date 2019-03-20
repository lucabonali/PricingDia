import matplotlib.pyplot as plt

class DemandCurve():
    def __init__(self, curve_values):
        self.curve_values = curve_values
        self.plot_phases_curve()


    '''
    Plot the demand curves of each phase
    '''
    def plot_phases_curve(self):
        plt.figure(0)
        plt.xlabel("Price")
        plt.ylabel("Demand")
        for i in range(len(self.curve_values)):
            plt.plot(self.curve_values[i][1], self.curve_values[i][0], linewidth=1, markersize=1)
        plt.show()
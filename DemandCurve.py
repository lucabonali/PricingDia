import matplotlib.pyplot as plt

class DemandCurve():
    def __init__(self, curve_values):
        self.curve_values = curve_values

    def plotCurve(self):
        plt.figure(0)
        plt.xlabel("Price")
        plt.ylabel("Demand")
        plt.plot(self.curve_values[0], self.curve_values[1], linewidth=1, markersize=1)
        plt.show()
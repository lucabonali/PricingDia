from Data import *
from MainPlot import *
import matplotlib as plt
import numpy as np

from NonStationaryEnvironment import *
from TS_Learner import *
from SWTS_Learner import *


n_arms = Data.n_arms

p_class1 = Data.get_class_probabilities(0)
p_class2 = Data.get_class_probabilities(1)
p_class3 = Data.get_class_probabilities(2)
p_agg = Data.get_class_probabilities(3)


t_horizon = Data.t_horizon
window_size = int(np.sqrt(t_horizon))




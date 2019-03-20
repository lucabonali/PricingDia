import numpy as np

#curve for the youngsters
first_curve_values = [[[1,0.9,0.9,0.8,0.7,0.6,0.6,0.5,0.1,0.05,0.05],[0,200,300,400,500,600,700,1000,1100,1200,1300]],        #launch phase feb, mar, apr
                      [[1,0.8,0.78,0.7,0.62,0.55,0.3,0.01,0.01,0.001,0.001],[0,200,300,400,500,600,700,1000,1100,1200,1300]], #new competitor's product set, ott, nov
                      [[1,0.8,0.78,0.7,0.62,0.55,0.44,0.1,0.05,0.001,0.001],[0,200,300,400,500,600,700,1000,1100,1200,1300]], #Holiday dic, gen
                      [[1,0.86,0.84,0.5,0.3,0.2,0.05,0.01,0.01,0.001,0.001],[0,200,300,400,500,600,700,1000,1100,1200,1300]]] #new model feb mar apr

#Curve for the adults
second_curve_values = [[[1,0.93,0.9,0.7,0.5,0.3,0.15,0.1,0.05,0.025,0.01],[0,200,300,400,500,550,650,700,800,900,1000]],
                       [[], [200,300,400,500,550,650,700,800,900,1000]],
                       [[], [200,300,400,500,550,650,700,800,900,1000]],
                       [[], [200,300,400,500,550,650,700,800,900,1000]]]

#Curve for the third age
third_curve_values = [[[0.5,0.5,0.35,0.3,0.2,0.1,0.05,0.03,0.05,0.1,0.05],[0,200,300,400,700,800,850,1000,1200,1300,1400]],
                      [[], [200,300,400,700,800,850,1000,1200,1300,1400]],
                      [[], [200,300,400,700,800,850,1000,1200,1300,1400]],
                      [[], [200,300,400,700,800,850,1000,1200,1300,1400]]]


cost_of_unit = 350
#mettere un parametro che si riferisce al numero di samples, in modo da provare un po' per quali valori è meglio l'aggragate e
#per qualiè meglio il disaggregate. Così da vedere quale usare nel caso in cui nel mondo realte ne possiamo ottenere tot al giorno
#bias variance tradeoff

import scipy.stats as scs


'''
Sequential K-testing
    - u = probability p of the candidate
    - v1 = sqrt(u*(1-u)/n)

    - H0: no change
    - H1: better change

https://towardsdatascience.com/the-math-behind-a-b-testing-with-example-code-part-1-of-2-7be752e1d06f

'''


class K_testing():
    def __init__(self, alpha, beta, delta):
        self.alpha = alpha #0,05
        self.beta = beta #0,8
        self.delta = delta #0,02

    '''
    Sequential A/B testing:
    loop all the selected candidates and compare them with the temporary best one,
    if the new one is better then it becomes the temporary best one
    '''
    def sequential_testing(self):
        best_candidate = self.candidates[0]
        for c in self.candidates:
            best_candidate = self.compare_candidates(best_candidate, c)
        return best_candidate

    '''
    Computation of the min number of candidates:
        - bcr(float): probability of success for control, sometimes referred to as baseline conversion rate
        - mde(float): minimum change in measurement between control group and test group if alternative
                      hypothesis is true, sometimes referred to as minimum detectable effect.
                      Our delta
    '''
    def minimum_sample_size(self, bcr):

        # standard normal distribution to determine z-values
        standard_norm = scs.norm(0, 1)

        # find Z_beta from desired power
        Z_beta = standard_norm.ppf(self.beta)

        # find Z_alpha
        Z_alpha = standard_norm.ppf(1 - self.alpha / 2)

        # average of probabilities from both groups
        pooled_prob = (bcr + bcr + self.delta) / 2
        sigma = 2 * pooled_prob * (1 - pooled_prob)

        return (sigma * (Z_beta + Z_alpha) ** 2) / self.delta ** 2


    def compare_candidates(self, a, b):
        better_one = a

        # Operatiooonnsssss for amazing comparison

        return better_one

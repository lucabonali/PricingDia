"""
SuperClass of NonStationaryEnvironment.
It interacts with the learner by returning a stochastic reward
"""
import Data
import numpy as np


class Environment:

    def __init__(self, n_arms, probabilities):
        """
        Initialization of the Environment:
        :param n_arms: number of candidates
        :param probabilities: probabilities of such candidates
        """
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.smooth_factor = Data.smooth_factor


    def round(self, pulled_arm):
        """
        Get the reward of the selected arm
        :param pulled_arm: the learner selected arm
        :return: conversion rate reward (taken from a Bernoulli distribution)
        """
        reward = np.random.binomial(1, self.probabilities[pulled_arm])

        return reward

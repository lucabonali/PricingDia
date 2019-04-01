"""
Non-stationary Environment.
The time horizon is divided into phases
"""

from Enviroment import *


class NonStationaryEnvironment(Environment):

    def __init__(self, n_arms, probabilities, horizon, samples_per_phase):
        """
        Initialization of the Non-stationary Environment:
        :param n_arms: number of candidates
        :param probabilities: probabilities of such candidates
        :param horizon: time horizon
        :param samples_per_phase: number of samples for each phase
        :self t: time initialization
        :self n_phases: number of phases in the time horizon
        :self cumulative_samples_per_phase: time breakpoints, when the phase changes
        """
        super().__init__(n_arms, probabilities)
        self.t = 0
        self.horizon = horizon
        self.phase_sizes = samples_per_phase
        self.n_phases = len(self.probabilities)
        self.cumulative_samples_per_phase = np.cumsum(self.phase_sizes)

    def get_current_phase(self, time):
        """
        Get the current phase
        :param time: the current time instant
        :return: the current phase
        """
        for i in range(self.n_phases):
            if time < self.cumulative_samples_per_phase[i]:
                return i
        return self.n_phases

    def round(self, pulled_arm):
        """
        Get the reward of the selected arm considering the current phase
        :param pulled_arm: the learner selected arm
        :return: conversion rate reward (taken from a Bernoulli distribution)
        """
        current_phase = self.get_current_phase(self.t)
        p = self.probabilities[current_phase][pulled_arm]
        self.t += 1

        reward = np.random.binomial(1, p)
        return reward

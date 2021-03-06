# Copyright (c) 2018, Raul Astudillo Marban
import time
import numpy as np

import numpy as np


class ParameterDistribution(object):
    """
    Class to handle the parameter distribution of the utility function.
    There are two possible ways to specify a parameter distribution: ...
    """

    def __init__(self, continuous=False, support=None, prob_dist=None, sample_generator=None):
        if continuous == True and sample_generator is None:
            pass
        else:
            self.continuous = continuous
            self.support = support
            self.prob_dist = prob_dist
            self.sample_generator = sample_generator

    def sample(self, number_of_samples=None):
        if self.continuous:
            parameter = self.sample_generator(number_of_samples)
        else:
            parameter = self.sample_discrete(support, prob_dist)
        return parameter

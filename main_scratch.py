import numpy as np
import functools
import operator
from utils import *
import scipy.ndimage.filters as filters
from itertools import product


class HarmonySearch(object):
    def __init__(self,
                 img,
                 iteration=1000,
                 sample_size=-1,
                 hmcr=0.7,
                 par=0.3,
                 adjust=0.5,
                 harmony_memory_size=40,
                 up_down_limit=None):
        super(HarmonySearch, self).__init__()
        self.iteration = iteration
        self.sample_size = sample_size
        self.hmcr = hmcr
        self.par = par
        self.adjust = adjust
        self.harmony_memory_size = harmony_memory_size
        self.hmm_matrix = list()  # Harmony matrix
        matrix = []  # Initializing empty matrix
        if up_down_limit == None:
            self.up_down_limit = [[0, 1]] * img.shape[0]
        else:
            self.up_down_limit = up_down_limit
        for limit in self.up_down_limit:
            row = np.random.uniform(low=limit[0], high=limit[1], size=(1, self.harmony_memory_size))[0]
            matrix.append(row)
        matrix = np.asarray(matrix).transpose()
        self.hmm_matrix = matrix

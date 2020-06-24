import numpy as np


class objective_function(object):

    def __init__(self,
                 input_Y,
                 iteration=5000,
                 weight_decimal=0,
                 sample_size=-1,
                 hmcr_proba=0.7,
                 par_proba=0.3,
                 adju_proba=0.5,
                 harmony_memory_size=40,
                 up_down_limit=None):
        super(objective_function, self).__init__()

        self.input_Y = input_Y
        self.iteration = iteration
        self.weight_decimal = weight_decimal
        if sample_size == -1:
            self.sample_size = len(input_Y)
        else:
            self.sample_size = sample_size
        self.hmcr_proba = hmcr_proba
        self.par_proba = par_proba
        self.adju_proba = adju_proba
        self.matrix_size = input_Y.shape
        self.harmony_memory_size = harmony_memory_size
        if up_down_limit == None:
            self.up_down_limit = [[[0, 0.1] for _ in range(input_Y.shape[0])] for _ in range(input_Y.shape[1])]
        else:
            self.up_down_limit = up_down_limit

    '''
    You should customize your fitness func here.
    '''

    def fitness(self, weight, target):
        e = 0.0
        for i in range(len(weight)):
            e+= np.mean(np.square((weight[i]) - (target)))
        e /= len(weight)
        return e

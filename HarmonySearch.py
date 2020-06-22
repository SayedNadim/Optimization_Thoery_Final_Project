import numpy as np
import functools
import operator
from utils import *


class HarmonyCore(object):

    def __init__(self, harmony_obj):
        self.obj_func = harmony_obj  # Objective function
        self.hmm_matrix = list() # Harmony matrix
        matrix = [] # Initializing empty matrix
        for limit in self.obj_func.up_down_limit:
            row = np.random.uniform(low=limit[0], high=limit[1], size=(1, self.obj_func.harmony_menmory_size))[0]
            matrix.append(row)
        matrix = np.asarray(matrix).transpose().round(self.obj_func.weight_decimal)
        # print(matrix)
        self.hmm_matrix = matrix
    def run(self):
        hmm_err_list = [0] * len(self.hmm_matrix) # Empty error list
        for m_i in range(len(self.hmm_matrix)):
            vetor_list = self.hmm_matrix[m_i]

            error = self.obj_func.fitness(vetor_list, self.obj_func.input_X, self.obj_func.input_Y)
            hmm_err_list[m_i] = error

        for itera in range(self.obj_func.iteration):
            vetor_list = [0] * self.obj_func.vector_size
            # while True:
            for i in range(self.obj_func.vector_size):
                if np.random.rand(1, )[0] < self.obj_func.hmcr_proba:
                    # new_vactor = 0.0
                    new_vactor = self.hmm_matrix[np.random.randint(self.obj_func.harmony_menmory_size, size=1)[0]][i]
                    if np.random.rand(1, )[0] < self.obj_func.par_proba:
                        if np.random.rand(1, )[0] < self.obj_func.adju_proba:
                            # new_vactor -= np.std(self.hmm_matrix[:][i]) * np.random.rand(1,)[0]
                            new_vactor -= (new_vactor - self.obj_func.up_down_limit[i][0]) * np.random.rand(1, )[0]
                        else:
                            # new_vactor += np.std(self.hmm_matrix[:][i]) * np.random.rand(1,)[0]
                            new_vactor += (self.obj_func.up_down_limit[i][0] - new_vactor) * np.random.rand(1, )[0]
                    vetor_list[i] = round(new_vactor, self.obj_func.weight_decimal)
                else:
                    new_vactor = \
                        np.random.uniform(low=self.obj_func.up_down_limit[i][0], high=self.obj_func.up_down_limit[i][1],
                                          size=(1,))[0]
                    vetor_list[i] = round(new_vactor, self.obj_func.weight_decimal)
                '''
                if not vetor_list in self.hmm_matrix:
                    break
                '''
            # Random Selection Sample
            if self.obj_func.sample_size > 0:
                random_idx = np.random.permutation(len(self.obj_func.input_X))[:self.obj_func.sample_size]
            else:
                random_idx = np.random.permutation(len(self.obj_func.input_X))
            error = self.obj_func.fitness(vetor_list, [self.obj_func.input_X[i] for i in random_idx],
                                          [self.obj_func.input_Y[i] for i in random_idx])
            overwrite_index = hmm_err_list.index(max(hmm_err_list))
            if hmm_err_list[overwrite_index] >= error:
                assert ([round(i, self.obj_func.weight_decimal) for i in vetor_list] == vetor_list)
                vetor_list = [round(i, self.obj_func.weight_decimal) for i in vetor_list]
                if not vetor_list in self.hmm_matrix.tolist():
                    # print('HMM_UPDATEP_NEW_VECTOR:',vetor_list)
                    # print('HMCR:',self.obj_func.hmcr_proba,'PAR:',self.obj_func.par_proba,'HMM_UPDATE_NEW_ERROR:',error)
                    hmm_err_list[overwrite_index] = error
                    self.hmm_matrix[overwrite_index] = vetor_list
            # else:
            #    print(vetor_list,'worst than',self.hmm_matrix[overwrite_index],'because new error:',error,'higher than',hmm_err_list[overwrite_index])
        return self.hmm_matrix, hmm_err_list, hmm_err_list.index(min(hmm_err_list)), error


class objective_function:

    def __init__(self,
                 input_X,
                 input_Y,
                 iteration=1000,
                 weight_decimal=0,
                 sample_size=-1,
                 hmcr_proba=0.7,
                 par_proba=0.3,
                 adju_proba=0.5,
                 harmony_menmory_size=40,
                 up_down_limit=None):

        self.input_X = input_X
        self.input_Y = input_Y
        self.iteration = iteration
        self.weight_decimal = weight_decimal
        if sample_size == -1:
            self.sample_size = len(input_X)
        else:
            self.sample_size = sample_size
        self.hmcr_proba = hmcr_proba
        self.par_proba = par_proba
        self.adju_proba = adju_proba
        self.vector_size = input_X.size
        self.harmony_menmory_size = harmony_menmory_size
        if up_down_limit == None:
            self.up_down_limit = [[-1, 1]] * input_X.size
        else:
            self.up_down_limit = up_down_limit

    '''
    You should customize your fitness func here.
    '''

    def fitness(self, weight, input_X, input_Y):
        e = 0.0
        # weight = [float(i) / sum(weight) for i in weight]
        mean = 0.5 * np.abs(np.mean(input_Y) - np.mean(input_X))
        std = 0.5 * np.abs(np.std(input_Y) - np.std(input_X))
        e += mean + std
        # for x, y in zip(input_X, input_Y):

        # e /= np.array(input_X).shape[0]
        # print(e)
        return e

    def error_patch(self, source, target):
        mean = 0.5 * np.abs(np.mean(source) - np.mean(target))
        std = 0.5 * np.abs(np.std(source) - np.std(target))
        return mean + std

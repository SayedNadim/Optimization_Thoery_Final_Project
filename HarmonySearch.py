import numpy as np
import functools
import operator
from utils import *


class HarmonyCore(object):

    def __init__(self, harmony_obj):
        self.obj_func = harmony_obj  # Objective function
        self.hmm_matrix = list()  # Harmony matrix
        matrix = []  # Initializing empty matrix
        for limit in self.obj_func.up_down_limit:
            row = np.random.uniform(low=limit[0], high=limit[1], size=(1, self.obj_func.harmony_memory_size))[0]
            matrix.append(row)
        matrix = np.asarray(matrix).transpose().round(self.obj_func.weight_decimal)
        # print(matrix)
        self.hmm_matrix = matrix

    def run(self):
        error = 0
        hmm_err_list = [0] * len(self.hmm_matrix)  # Empty error list
        for m_i in range(len(self.hmm_matrix)):
            vector_list = self.hmm_matrix[m_i]

            error = self.obj_func.fitness(vector_list, self.obj_func.input_X, self.obj_func.input_Y)
            hmm_err_list[m_i] = error

        for itera in range(self.obj_func.iteration):
            vector_list = [0] * self.obj_func.vector_size
            # while True:
            for i in range(self.obj_func.vector_size):
                if np.random.rand(1, )[0] < self.obj_func.hmcr_proba:
                    # new_vector = 0.0
                    new_vector = self.hmm_matrix[np.random.randint(self.obj_func.harmony_memory_size, size=1)[0]][i]
                    if np.random.rand(1, )[0] < self.obj_func.par_proba:
                        if np.random.rand(1, )[0] < self.obj_func.adju_proba:
                            # new_vector -= np.std(self.hmm_matrix[:][i]) * np.random.rand(1,)[0]
                            new_vector -= (new_vector - self.obj_func.up_down_limit[i][0]) * np.random.rand(1, )[0]
                        else:
                            # new_vector += np.std(self.hmm_matrix[:][i]) * np.random.rand(1,)[0]
                            new_vector += (self.obj_func.up_down_limit[i][0] - new_vector) * np.random.rand(1, )[0]
                    vector_list[i] = round(new_vector, self.obj_func.weight_decimal)
                else:
                    new_vector = \
                        np.random.uniform(low=self.obj_func.up_down_limit[i][0], high=self.obj_func.up_down_limit[i][1],
                                          size=(1,))[0]
                    vector_list[i] = round(new_vector, self.obj_func.weight_decimal)
                '''
                if not vector_list in self.hmm_matrix:
                    break
                '''
            # Random Selection Sample
            if self.obj_func.sample_size > 0:
                random_idx = np.random.permutation(len(self.obj_func.input_X))[:self.obj_func.sample_size]
            else:
                random_idx = np.random.permutation(len(self.obj_func.input_X))
            error = self.obj_func.fitness(vector_list, [self.obj_func.input_X[i] for i in random_idx],
                                          [self.obj_func.input_Y[i] for i in random_idx])
            overwrite_index = hmm_err_list.index(max(hmm_err_list))
            if hmm_err_list[overwrite_index] >= error:
                assert ([round(i, self.obj_func.weight_decimal) for i in vector_list] == vector_list)
                vector_list = [round(i, self.obj_func.weight_decimal) for i in vector_list]
                if not vector_list in self.hmm_matrix.tolist():
                    # print('HMM_UPDATEP_NEW_VECTOR:',vector_list)
                    # print('HMCR:',self.obj_func.hmcr_proba,'PAR:',self.obj_func.par_proba,'HMM_UPDATE_NEW_ERROR:',error)
                    hmm_err_list[overwrite_index] = error
                    self.hmm_matrix[overwrite_index] = vector_list
            # else:
            #    print(vector_list,'worst than',self.hmm_matrix[overwrite_index],'because new error:',error,'higher than',hmm_err_list[overwrite_index])
        return self.hmm_matrix, hmm_err_list, hmm_err_list.index(min(hmm_err_list)), error

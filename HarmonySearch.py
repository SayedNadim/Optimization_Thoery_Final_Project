import numpy as np
import functools
import operator
from utils import *


class HarmonyCore(object):

    def __init__(self, harmony_obj):
        self.obj_func = harmony_obj  # Objective function
        self.hmm_matrix = list()  # Harmony matrix
        self.matrix_size = self.obj_func.matrix_size
        matrix = np.random.uniform(0, 0.5, size=(self.matrix_size[0], self.matrix_size[1], self.obj_func.harmony_memory_size))
        matrix = np.asarray(matrix).transpose()
        self.hmm_matrix = matrix

    def run(self):
        error = 0
        hmm_err_list = [0] * len(self.hmm_matrix)  # Empty error list
        # print(np.asarray(hmm_err_list).shape)
        for m_i in range(len(self.hmm_matrix)):
            # print(len(self.hmm_matrix))
            matrix_list = self.hmm_matrix[m_i]
            error = self.obj_func.fitness(matrix_list, self.obj_func.input_Y)
            hmm_err_list[m_i] = error

        for itera in range(self.obj_func.iteration):
            matrix_list = [[0 for _ in range(self.matrix_size[0])] for _ in range(self.matrix_size[1])]
            for i in range(self.obj_func.matrix_size[0]):
                for j in range(self.obj_func.matrix_size[1]):
                    if np.random.rand(1, )[0] < self.obj_func.hmcr_proba:
                        new_matrix = \
                            self.hmm_matrix[np.random.randint(self.obj_func.harmony_memory_size, size=1)[0]][i][j]
                        if np.random.rand(1, )[0] < self.obj_func.par_proba:
                            if np.random.rand(1, )[0] < self.obj_func.adju_proba:
                                new_matrix -= (new_matrix - self.obj_func.up_down_limit[i][j][0]) * np.random.rand(1, )[0]
                            else:
                                new_matrix += (self.obj_func.up_down_limit[i][j][0] - new_matrix) * np.random.rand(1, )[0]
                        matrix_list[i][j] = new_matrix
                    else:
                        new_matrix = \
                            np.random.uniform(low=self.obj_func.up_down_limit[i][j][0],
                                              high=self.obj_func.up_down_limit[i][j][1],
                                              size=(1,))[0]
                        matrix_list[i][j] = new_matrix

            # Random Selection Sample
            if self.obj_func.sample_size > 0:
                random_idx = np.random.permutation(len(matrix_list))[:self.obj_func.sample_size]
            else:
                random_idx = np.random.permutation(len(matrix_list))
            error = self.obj_func.fitness([matrix_list[i] for i in random_idx], self.obj_func.input_Y)
            overwrite_index = hmm_err_list.index(max(hmm_err_list))
            if hmm_err_list[overwrite_index] >= error:
                for i in range(np.asarray(matrix_list).shape[0]):
                    for j in range(np.asarray(matrix_list).shape[1]):
                        matrix_list[i][j] = matrix_list[i][j]
                if not matrix_list in self.hmm_matrix.tolist():
                    hmm_err_list[overwrite_index] = error
                    self.hmm_matrix[overwrite_index] = matrix_list
        return self.hmm_matrix, hmm_err_list, hmm_err_list.index(min(hmm_err_list)), error

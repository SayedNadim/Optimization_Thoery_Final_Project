import numpy as np
from HarmonySearch import HarmonyCore
from objective_function import objective_function
from utils import *


def harmonyfunction(y_gray_patches, y_color_patches, iteration):
    error = 0
    for j in range(y_gray_patches.shape[0]):
        for k in range(y_gray_patches.shape[1]):
            of = objective_function(y_gray_patches[j][k], y_color_patches[j][k], sample_size=6)
            hs = HarmonyCore(of)
            hmm_vector, hmm_err_list, err_idx, err = hs.run()
            # print('HMM_Vector:', hmm_vector)
            # print('Best HMM_Vector:', hmm_vector[err_idx])
            # print('hmm_err_list:', hmm_err_list)
            # print('err_index:', err_idx, 'hmm_err:', hmm_err_list[err_idx])
            error += err
            print('%d th population, %d%d patch error: %0.5f, total error: %0.5f' % (iteration, j, k, err, error))
            source_patches.append(y_gray_patches[j][k])
            target_patches.append(y_color_patches[j][k])
    return source_patches, target_patches, error


image_gray, y_gray, i_gray, q_gray = read_image('tiger.jpg')
image_color, y_color, i_color, q_color = read_image('tiger_color.jpg')

# cv2.imshow('i_gray', i_gray)
# cv2.imshow('i_color', i_color)
# cv2.waitKey(0)
#


#
y_gray_patches = create_patches(y_gray, 15, 15)
y_color_patches = create_patches(y_color, 15, 15)
# print(y_color_patches.shape)
# 
# 

population = 1
optimizatin_error = []
source_patches = []
target_patches = []
error_run = []
for i in range(population):
    source, target, error = harmonyfunction(y_gray_patches, y_color_patches, i)
    source_patches.append(source)
    target_patches.append(target)
    error_run.append(error)

low_error = minimum_index(error_run)

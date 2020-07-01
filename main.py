import numpy as np
from HarmonySearch import HarmonyCore
from objective_function import objective_function
from utils import *
import time


def harmonyfunction(y_color_patches, pop, i_or_q='i'):
    error = 0
    of = objective_function(y_color_patches, sample_size=6)
    hs = HarmonyCore(of)
    hmm_vector, hmm_err_list, err_idx, err = hs.run()
    error += err
    best = hmm_vector[err_idx]
    if i_or_q == 'i':
        print('i color space - %d th population,  error: %0.5f' % (pop+1, err))
    elif i_or_q == 'q':
        print('q color space - %d th population,  error: %0.5f' % (pop+1, err))
    return best, error


def main():
    size = (32, 32)
    image_gray, y_gray, i_gray, q_gray = read_image('images/flower_gray_32.jpeg', size=size)
    image_color, y_color, i_color, q_color = read_image('images/flower_gray_32.jpeg', size=size)

    total_run = 10
    error_i = []
    error_q = []
    i_set = []
    q_set = []
    start = time.time()
    for i in range(total_run):
        i_c, e_i = harmonyfunction(i_color, i, i_or_q='i')
        q_c, e_q = harmonyfunction(q_color, i, i_or_q='q')
        i_set.append(i_c)
        q_set.append(q_c)
        error_i.append(e_i)
        error_q.append(e_q)

    min_i, min_error_i = minimum_index(error_i)
    # print(min_error_i, min_i)
    min_q, min_error_q = minimum_index(error_q)
    # print(min_error_q, min_q)

    # i_error = np.mean(i_color - i_set[min_i])
    # q_error = np.mean(q_color - q_set[min_q])
    # print(i_error, q_error)
    print(i_color)
    print('#'*80)
    print(i_set[min_i])
    print('*'*80)
    print(q_color)
    print('#'*80)
    print(q_set[min_q])
    print('*'*80)

    result = cv2.merge((y_gray.astype(np.float32), i_set[min_i].astype(np.float32), q_set[min_q].astype(np.float32)))
    result = cv2.cvtColor(result, cv2.COLOR_YUV2BGR)
    mean_error = np.mean(np.abs(image_color - result))
    print("mean error; {}".format(mean_error * 100))
    end = time.time()
    print("total time: {}".format(end - start))

    cv2.imwrite('images/flower_colored.png', (result * 255).astype(np.uint))
    # import matplotlib.pyplot as plt
    # plt.imshow((result * 255.0).astype(np.int), aspect='auto')
    # plt.show()

if __name__ == '__main__':
    main()
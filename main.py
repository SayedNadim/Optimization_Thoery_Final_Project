from HarmonySearch import HarmonyCore, objective_function
from utils import *

patch_size = 3
x_im = imageio.imread('tiger.jpg')
x_hint = imageio.imread('tiger_color.jpg')
x_im = np.resize(x_im, (256, 256, 3))
x_hint = np.resize(x_hint, (256, 256, 3))

patch_size = 3
yiq_gray, yiq_color = color_space_conversion(x_im, x_hint)
M = x_im.shape[0] // patch_size
N = x_im.shape[1] // patch_size
grid = np.zeros((M, N))
yiq_gray_luminance = yiq_gray[:, :, 0]
yiq_color_luminance = yiq_color[:, :, 0]
yiq_gray_patches = create_patches(yiq_gray_luminance, patch_size, patch_size)
yiq_color_patches = create_patches(yiq_color_luminance, patch_size, patch_size)

gray_array =[]
color_array = []

for i in range(M):
    for j in range(N):
        grays = img2vector(yiq_gray_patches[i][j])
        colors = img2vector(yiq_color_patches[i][j])
        gray_array.append(grays)
        color_array.append(colors)



population = []
for i in range(len(gray_array)):
    of = objective_function(gray_array[i], color_array[i], sample_size=6, weight_decimal=2, )
    hs = HarmonyCore(of)
    '''
    run harmony
    hs.run()
    '''
    hmm_vector, hmm_err_list, err_idx = hs.run()
    population.append(hmm_vector[err_idx])

    '''
    then you can get (hmm_vector,hmm_err_list,err_idx) after hs return.
    '''

    # print('HMM_Vector:', hmm_vector)
    print('Best HMM_Vector:', hmm_vector[err_idx])
    print('hmm_err_list:', hmm_err_list)
    print('err_index:', err_idx, 'hmm_err:', hmm_err_list[err_idx])
    print('{}th patch done'.format(i))



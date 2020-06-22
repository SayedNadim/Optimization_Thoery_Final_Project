from HarmonySearch import HarmonyCore, objective_function
from utils import *

patch_size = 3
x_im = imageio.imread('tiger.jpg')
x_hint = imageio.imread('tiger_color.jpg')
x_im = np.resize(x_im, (32, 32, 3))
x_hint = np.resize(x_hint, (32, 32, 3))
yiq_gray, yiq_color = color_space_conversion(x_im, x_hint)
M = x_im.shape[0] // patch_size
N = x_im.shape[1] // patch_size
yiq_gray_luminance = yiq_gray[:, :, 0]
yiq_color_luminance = yiq_color[:, :, 0]
yiq_gray_patches = create_patches(yiq_gray_luminance, patch_size, patch_size)
yiq_color_patches = create_patches(yiq_color_luminance, patch_size, patch_size)

gray_array = []
color_array = []

# for i in range(M):
#     for j in range(N):
#         grays = img2vector(yiq_gray_patches[i][j])
#         colors = img2vector(yiq_color_patches[i][j])
#         gray_array.append(grays)
#         color_array.append(colors)
#
# print(len(gray_array))

best_values = []
for i in range(M):
    for j in range(N):
        of = objective_function(yiq_gray_patches[i][j], yiq_color_patches[i][j], sample_size=6, weight_decimal=2, )
        hs = HarmonyCore(of)
        '''
        run harmony
        hs.run()
        '''
        hmm_vector, hmm_err_list, err_idx = hs.run()
        best_values.append(hmm_vector[err_idx])

        '''
        then you can get (hmm_vector,hmm_err_list,err_idx) after hs return.
        '''
        # print('HMM_Vector:', hmm_vector)
        print('Best HMM_Vector:', hmm_vector[err_idx])
        # print('hmm_err_list:', hmm_err_list)
        # print('err_index:', err_idx, 'hmm_err:', hmm_err_list[err_idx])
        print('{}{}th patch done'.format(i, j))

best_values = np.reshape(best_values, (M, N, patch_size, patch_size))
print(best_values)
final_image = reconstruct_image(best_values)
print(final_image.shape)
import matplotlib.pyplot as plt
plt.imshow(final_image, aspect='auto')
plt.show()
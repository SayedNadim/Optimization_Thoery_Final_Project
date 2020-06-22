import numpy as np
import imageio
from utils import create_patches, reconstruct_image, image_preprocess, color_space_conversion, error_patch, patch_center

patch_size = 5
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

genes = []
patches = []
coordinates = []
errors = []
for m in range(M):
    for n in range(N):
        target_patches = yiq_gray_patches[m][n]
        source_coordinate = patch_center(yiq_gray_patches[m][n])
        error = error_patch(yiq_color_patches[m][n], yiq_gray_patches[m][n])
        patches.append(target_patches)
        coordinates.append(source_coordinate)
        errors.append(error)

genes.append(patches)
genes.append(coordinates)
genes.append(sum(errors))

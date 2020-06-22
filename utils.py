import numpy as np
import functools
import operator
import colorsys
import imageio
import skimage
import cv2
from skimage import io, util


def patch_center(a):
    x, y = a.shape[0], a.shape[1]
    return ((x-1)/2,(y-1)/2)

def img2vector(image):
    """
    Converts image into 1D vector
    :param image: Image
    :return: 1D vector
    """
    return np.reshape(a=image, newshape=(functools.reduce(operator.mul, image.shape)))


def vecotr2img(vector, shape):
    """
    Converts a 1D vector to array
    :param vector: input vector
    :param shape: shape of the image
    :return: image
    """
    # check if the vector can be represented according to the specific shape
    if len(vector) != functools.reduce(operator.mul, shape):
        raise ValueError(
            'A vector of length {vector_length} into an array of shape {shape}'.format(vector_length=len(vector),
                                                                                       shape=shape))
    return np.reshape(a=vector, newshape=shape)


def yiq_rgb(y, i, q):
    """
    Takes Y, I and Q channels and returns converted R, G and B channels
    :param y: Y channel of the image
    :param i: I channel of the image
    :param q: Q channel of the image
    :return: R, G and B channels
    """
    r_raw = y + 0.948262 * i + 0.624013 * q
    g_raw = y - 0.276066 * i - 0.639810 * q
    b_raw = y - 1.105450 * i + 1.729860 * q
    r_raw[r_raw < 0] = 0
    r_raw[r_raw > 1] = 1
    g_raw[g_raw < 0] = 0
    g_raw[g_raw > 1] = 1
    b_raw[b_raw < 0] = 0
    b_raw[b_raw > 1] = 1
    return (r_raw, g_raw, b_raw)


def image_preprocess(original, hinted_image):
    """
    Takes original as well as hinted image and performs colorspace conversion.
    :param original: Original grayscale image
    :param hinted_image: Grayscale image with user scrabbles
    :return: Difference image and YIQ (YUV) colorspace image
    """
    original = original.astype(float) / 255
    hinted_image = hinted_image.astype(float) / 255
    colorIm = abs(original - hinted_image).sum(2) > 0.01
    (Y, _, _) = colorsys.rgb_to_yiq(original[:, :, 0], original[:, :, 1], original[:, :, 2])
    (_, I, Q) = colorsys.rgb_to_yiq(hinted_image[:, :, 0], hinted_image[:, :, 1], hinted_image[:, :, 2])
    ntscIm = np.zeros(original.shape)
    ntscIm[:, :, 0] = Y
    ntscIm[:, :, 1] = I
    ntscIm[:, :, 2] = Q
    return colorIm, ntscIm



def color_space_conversion(gray, color):
    gray = gray.astype(float) / 255
    color = color.astype(float) / 255
    (Y_gray, _, _) = colorsys.rgb_to_yiq(gray[:, :, 0], gray[:, :, 1], gray[:, :, 2])
    (_, I_gray, Q_gray) = colorsys.rgb_to_yiq(gray[:, :, 0], gray[:, :, 1], gray[:, :, 2])
    (Y_color, _, _) = colorsys.rgb_to_yiq(color[:, :, 0], color[:, :, 1], color[:, :, 2])
    (_, I_color, Q_color) = colorsys.rgb_to_yiq(color[:, :, 0], color[:, :, 1], color[:, :, 2])
    yiq_gray = np.zeros(gray.shape)
    yiq_color = np.zeros(color.shape)
    yiq_gray[:, :, 0] = Y_gray
    yiq_gray[:, :, 1] = I_gray
    yiq_gray[:, :, 2] = Q_gray
    yiq_color[:, :, 0] = Y_color
    yiq_color[:, :, 1] = I_color
    yiq_color[:, :, 2] = Q_color
    return yiq_gray, yiq_color


def error_patch(target, source):
    mean = 0.5 * np.abs(np.mean(source) - np.mean(target))
    std = 0.5 * np.abs(np.std(source) - np.std(target))
    return mean + std


def create_patches(img, patch_width, patch_height):
    # Trim right and bottom borders if image size is not
    # an integer multiple of patch size
    nrows, ncols = patch_height, patch_width
    trimmed = img[:img.shape[0] // nrows * nrows, :img.shape[1] // ncols * ncols]

    # # Create folder to store results if necessary
    # patch_dir = os.path.join(folder, 'Patched Image')
    # if not os.path.isdir(patch_dir):
    #     os.mkdir(patch_dir)

    # Generate patches and save them to disk
    patches = util.view_as_blocks(trimmed, (nrows, ncols))
    # for i in range(patches.shape[0]):
    #     for j in range(patches.shape[1]):
    #         patch = patches[i, j, :, :]
    #         patch_name = f'patch_{i:02}_{j:02}.png'
    #         io.imsave(os.path.join(patch_dir, patch_name), patch)

    return patches


def reconstruct_image(patches):
    img_height = patches.shape[0] * patches.shape[2]
    img_width = patches.shape[1] * patches.shape[3]
    return patches.transpose(0, 2, 1, 3).reshape(img_height, img_width)

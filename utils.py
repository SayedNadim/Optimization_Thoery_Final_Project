import numpy as np
import functools
import operator
import colorsys
import imageio


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


def error_patch(target, source):
    mean = 0.5 * np.abs(np.mean(source) - np.mean(target))
    std = 0.5 * np.abs(np.std(source) - np.std(target))
    return mean + std


def padarray(A, patch_size):
    t2 = A.shape[0] % patch_size[1]
    return np.pad(A, pad_width=(0, t2), mode='constant')


def extract_patches(a, patch_size):
    sz = a.itemsize
    h1, w1 = a.shape
    if h1 % patch_size[0] != 0 or w1 % patch_size[1] != 0:
        a = padarray(a, patch_size)
    h, w = a.shape
    bh, bw = patch_size
    shape = (h // bh, w // bw, bh, bw)
    strides = sz * np.array([w * bh, bw, w, 1])
    blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return blocks




if __name__ == '__main__':
    patch_size = 3
    x_im = imageio.imread('baby.bmp')
    x_hint = imageio.imread('baby_marked.bmp')
    colorImage, ntscImage = image_preprocess(x_im, x_hint)
    print(colorImage.shape, ntscImage.shape)

    M = x_im.shape[0] // patch_size
    N = x_im.shape[1] // patch_size
    # grid = np.arange(M * N).reshape((M, N))
    # print(grid.shape)
    # print(grid)
    luminance_image = ntscImage[:, :, 0]
    print(luminance_image.shape)
    luminance_image_patches = extract_patches(luminance_image, (3,3))
    print(luminance_image_patches[0][0])
    hint_image_patches = extract_patches(colorImage, (3,3))
    print(hint_image_patches[0][0])
    grid = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            grid[i][j] = error_patch(luminance_image_patches[i][j], hint_image_patches[i][j] )
    print(grid[0][0])
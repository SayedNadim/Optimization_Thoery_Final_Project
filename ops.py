import numpy as np


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    result = (arr.reshape(h // nrows, nrows, -1, ncols)
              .swapaxes(1, 2)
              .reshape(-1, nrows, ncols))
    return result


image = np.random.rand(256, 256)
# print(image.shape)
# ph, pw = 3, 3
#
# patches = []
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         patch = image[i:i+ph, j:j+pw]
#         patches.append(patch)
#
#
# print(image)
# print(patches)

patches = blockshaped(image, 4, 4)
print(image)
print(patches)

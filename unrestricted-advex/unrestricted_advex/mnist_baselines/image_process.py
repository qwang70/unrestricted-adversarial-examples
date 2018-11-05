import numpy as np
from scipy import ndimage

"""
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train.shape)
for i in range(x_train.shape[0]):
    img = x_train[i]
    blur = ndimage.gaussian_filter(img, sigma=5)
    x_train[i] = blur
"""
def apply_gaussian_filter(imgaes, sigma = 5):
    for i in range(images.shape[0]):
        img_2d = np.reshape(images[i], (28,28))
        blur = ndimage.gaussian_filter(img_2d, sigma=sigma)
        images[i] = np.reshape(blur, (-1))
    return images

def apply_gaussian_filter_3d(images, sigma = 5):
    for i in range(images.shape[0]):
        blur = ndimage.gaussian_filter(images[i], sigma=sigma)
        images[i] = blur
    return images

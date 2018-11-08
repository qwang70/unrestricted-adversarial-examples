import numpy as np
from scipy import ndimage
import collections
from tensorflow.contrib.learn.python.learn.datasets import mnist

"""
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train.shape)
for i in range(x_train.shape[0]):
    img = x_train[i]
    blur = ndimage.gaussian_filter(img, sigma=5)
    x_train[i] = blur
"""
def apply_gaussian_to_dataset(dataset):
    blur_dataset = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    train_images = apply_gaussian_filter(dataset.train.images)
    test_images = apply_gaussian_filter(dataset.test.images)
    validation_images = apply_gaussian_filter(dataset.validation.images)

    blur_dataset.train = mnist.DataSet(train_images, dataset.train.labels, reshape=False)
    blur_dataset.test = mnist.DataSet(test_images, dataset.test.labels, reshape=False)
    blur_dataset.validation = mnist.DataSet(validation_images, dataset.validation.labels, reshape=False)

    return blur_dataset

def apply_gaussian_filter(images, sigma = 2, scale_to_255 = True):
    for i in range(images.shape[0]):
        img_2d = np.reshape(images[i], (28,28))
        blur = ndimage.gaussian_filter(img_2d, sigma=sigma)
        if scale_to_255:
            blur *= 255
        images[i] = np.reshape(blur, (-1))
    return images

def map_gaussian_filter(image, sigma = 2):
    img_2d = np.reshape(image, (28,28))
    blur = ndimage.gaussian_filter(img_2d, sigma=sigma)
    image = np.reshape(blur, (-1))
    return image


def apply_gaussian_filter_3d(images, sigma = 2):
    for i in range(images.shape[0]):
        blur = ndimage.gaussian_filter(images[i], sigma=sigma)
        images[i] = blur
    return images

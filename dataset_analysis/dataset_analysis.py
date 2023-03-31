import numpy as np

def max_reflectance_train(dataset):
    '''Find maximum reflectance in train set'''
    max = 0
    for image, label, height, width in dataset:
        max_reflectance = np.amax(image)
        if max_reflectance > max:
            max = max_reflectance
    return max

def min_reflectance_train(dataset):
    '''Find minimum reflectance in train set'''
    min = 1e+6
    for image, label, height, width in dataset:
        min_reflectance = np.amin(image)
        if min_reflectance < min:
            min = min_reflectance
    return min

def max_reflectance_test(dataset):
    '''Find maximum reflectance in test set'''
    max = 0
    for filename, image, height, width in dataset:
        max_reflectance = np.amax(image)
        if max_reflectance > max:
            max = max_reflectance
    return max

def min_reflectance_test(dataset):
    '''Find minimum reflectance in test set'''
    min = 1e+6
    for filename, image, height, width in dataset:
        min_reflectance = np.amin(image)
        if min_reflectance < min:
            min = min_reflectance
    return min
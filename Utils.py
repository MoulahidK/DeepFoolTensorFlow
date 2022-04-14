import tensorflow
import tensorflow.keras
import tensorflow.keras.backend as K
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *

from tensorflow.keras.models import *
from tensorflow.python.ops import nn


import glob
import skimage.io as io
import random

def hermitePolynomials(order, x, sigma):

    res = tensorflow.math.pow((np.sqrt(2)/sigma)* x,order)
    for i in range(1, (order//2)+1):
        term = (tensorflow.math.multiply(math.pow(-1, i)*math.factorial(order)/(math.factorial(i)*math.factorial(order - 2*i)), tensorflow.math.pow((tensorflow.math.divide(np.sqrt(2),sigma)),(order - 2*i)))) * tensorflow.math.pow(x, (order - 2*i))
        res = tensorflow.math.add(res, term)
    return res


def computeGaussianDerivative(order, x, sigma):
    
    hermitePart = tensorflow.math.multiply(tensorflow.math.pow(tensorflow.math.divide(-1,tensorflow.math.multiply(math.sqrt(2),sigma)),order), hermitePolynomials(order, x, sigma))
    gaussianPart = tensorflow.math.multiply(tensorflow.math.divide(1,tensorflow.math.multiply(sigma,np.sqrt(2*np.pi))), tensorflow.math.exp(- tensorflow.math.divide(tensorflow.math.pow(x, 2),(2*tensorflow.math.pow(sigma,2)))))
    
    gaussianDerivative = tensorflow.math.multiply(hermitePart, gaussianPart)
    return gaussianDerivative


def computeGaussianBasis(size, order, sigmas, centroids, thetas):

    kernels = []
    [x,y] = tensorflow.meshgrid(range(-int(size[0]/2), int(size[0]/2) + 1), range(-int(size[1]/2), int(size[1]/2) + 1))
    x = tensorflow.cast(x, tensorflow.float32)
    y = tensorflow.cast(y, tensorflow.float32)
    counter = 0
    for i in range(order+1):
        for j in range(i+1):

            u = tensorflow.math.add(tensorflow.multiply(tensorflow.math.cos(thetas[counter]), x), tensorflow.math.multiply(tensorflow.math.sin(thetas[counter]), y))
            v = tensorflow.math.add(tensorflow.multiply(-tensorflow.math.sin(thetas[counter]), x), tensorflow.math.multiply(tensorflow.math.cos(thetas[counter]), y))
        
            dGaussx = computeGaussianDerivative(j, tensorflow.math.add(u, - centroids[counter, 0]), sigmas[counter, 0])
            dGaussy = computeGaussianDerivative(i-j, tensorflow.math.add(v, - centroids[counter, 1]), sigmas[counter, 1])
            
            dGauss = tensorflow.math.multiply(dGaussx, dGaussy)
            kernels.append(tensorflow.expand_dims(dGauss, -1))
            counter += 1
    return tensorflow.stack(kernels, axis = -1)


def getBasis(size, numBasis, order, sigmas, centroids, thetas):
    
    basis = []
    for i in range(numBasis):
        basis.append(computeGaussianBasis(size, order, sigmas[i,:,:], centroids[i,:,:], thetas[i,:]))
        
    return tensorflow.stack(basis, axis = 0)

def getGaussianFilters(basis, weights, numBasis, inputChannels, outputChannels):
    
    Filters = []
    for i in range(0, numBasis):

        consideredBasis = basis[i,:,:,:,:]
        consideredBasis = tensorflow.expand_dims(consideredBasis, axis = -2)
        consideredBasis = tensorflow.tile(consideredBasis, [1,1, inputChannels, int(outputChannels/numBasis), 1])
        filters = tensorflow.multiply(consideredBasis, weights[i,:,:,:])
        filters = tensorflow.reduce_sum(filters, axis = -1)
        Filters.append(filters)
    Filters = tensorflow.concat(Filters, axis = -1)
    return Filters
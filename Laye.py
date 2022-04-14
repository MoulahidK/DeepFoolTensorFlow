from Utils import *

class FTGDConvLayerInference(tensorflow.keras.layers.Layer):

    def __init__(self, filters, kernel_size, order, num_basis, sigmas, thetas, centroids, CLweights, bias, padding, strides, **kwargs):

        super(FTGDConvLayerInference, self).__init__()
        self.num_filters = filters
        self.filter_size = kernel_size
        self.numBasis = num_basis
        self.order = order
        self.paddingMode = padding
        self.stride = strides
        self.sigmas = tensorflow.Variable(np.array(sigmas), dtype = 'float', name='Sigmas')
        self.thetas = tensorflow.Variable(np.array(thetas), dtype = 'float', name='Thetas')
        self.centroids = tensorflow.Variable(np.array(centroids), dtype = 'float', name='Centroids')
        self.CLweights = tensorflow.Variable(np.array(CLweights), dtype = 'float', name='CLWeights')
        self.bias = tensorflow.Variable(np.array(bias), dtype = 'float', name = 'Bias')

    def build(self, input_shape):

        self.inputChannels = input_shape[-1]
        basis = getBasis(self.filter_size, self.numBasis, self.order, self.sigmas, self.centroids, self.thetas)
        self.GaussianFilters = getGaussianFilters(basis, self.CLweights, self.numBasis, self.inputChannels, self.num_filters)

    def call(self, inputs):

        output = K.conv2d(inputs, self.GaussianFilters, strides = self.stride, padding = self.paddingMode)
        output = tensorflow.math.add(output, self.bias)
        return output


    def get_config(self):
        config = super(FTGDConvLayerInference, self).get_config()
        config.update({
            "filters":self.num_filters,
            "kernel_size":self.filter_size,
            'order':self.order,
            'num_basis':self.numBasis,
            'sigmas':self.sigmas.numpy(),
            'thetas':self.thetas.numpy(),
            'centroids':self.centroids.numpy(),
            'CLweights':self.CLweights.numpy(),
            'bias':self.bias.numpy(),
            'padding':self.paddingMode,
            'strides':self.stride, 
        })
        return config

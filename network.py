import numpy as np
import util


class BackPropNetwork:
    def __init__(self, inp):
        inp = np.array(inp)
        self._input = inp
        self._constructed = False
        self._layers = [inp.shape[0]+1]
        self._activations = []
        self._weights = []
        self._forward_cache_acted = []
        self._forward_cache_raw = []
        self._gradients = []
        self._predictions = None
        self._labels = None

    def add_layer(self, units, activation=None):
        if self._constructed:
            print("Cannot add layers after the network is built")
            assert 0
        self._layers.append(units)
        self._activations.append(activation)

    def build(self, labels):
        self._labels = np.array(labels)
        layers = len(self._layers)
        for i in range(layers - 1):
            weight = np.random.randn(self._layers[i+1], self._layers[i])  # a constant value
            # weight = np.ones((self._layers[i+1], self._layers[i]))
            self._weights.append(weight)
        self._constructed = True

    def forward(self):
        temp = np.vstack((np.ones((1, self._input.shape[1])), self._input))
        self._forward_cache_acted = [temp]
        self._forward_cache_raw = [temp]
        if not self._constructed:
            print("use the build method before forwarding.")
            assert 0
        times = len(self._layers) - 1
        for i in range(times):
            # temp = np.vstack((np.ones((1, self._input.shape[1])), temp))
            temp = np.dot(self._weights[i], temp)
            self._forward_cache_raw.append(temp)
            if not self._activations[i]:
                pass
            elif self._activations[i].lower() == 'sigmoid':
                temp = util.sigmoid(temp)
            elif self._activations[i].lower() == 'tanh':
                temp = util.tanh(temp)
            elif self._activations[i].lower() == 'relu':
                temp = util.relu(temp)
            else:
                print("Activation function should be None, 'sigmoid', 'tanh' or 'relu'.")
                assert 0
            self._forward_cache_acted.append(temp)

        self._predictions = temp
        return temp

    def backward(self, learning_rate=0.01):
        # using mse for loss
        self._gradients = []
        mse = np.average(np.square(self._forward_cache_acted[-1] - self._labels))
        d_mse_yhat = np.average(2 * (self._forward_cache_acted[-1] - self._labels))
        times = len(self._layers) - 1
        dx = np.ones((self._forward_cache_raw[times-2].shape[0], 1))
        for i in range(times-1, -1, -1):
            # in reverse order
            act = self._activations[i]
            d_act = None
            if not act:
                d_act = np.ones(self._forward_cache_raw[i+1].shape)
                # d_act = 1
            elif act.lower() == 'sigmoid':
                d_act = util.sigmoid(self._forward_cache_raw[i+1]) * (1 - util.sigmoid(self._forward_cache_raw[i+1]))
            elif act.lower() == 'relu':
                d_act = (self._forward_cache_raw[i+1] > 0).astype('float32')
            elif act.lower() == 'tanh':
                d_act = 1 - np.square(util.tanh(self._forward_cache_raw[i+1]))
            if i != times-1:
                dw = np.dot(dx * d_act, self._forward_cache_raw[i].T) / self._labels.shape[1]
            else:
                dw = np.dot(d_act, self._forward_cache_raw[i].T) / self._labels.shape[1]

            dx = np.dot(d_act.T, self._weights[i]).T
            self._gradients.insert(0, dw * d_mse_yhat)
        for i in range(times):
            self._weights[i] = self._weights[i] - learning_rate * self._gradients[i]
        return mse







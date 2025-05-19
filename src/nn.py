import numpy as np

from src.Engine import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x):
        act = sum(xi * wi for xi, wi in zip(x, self.w)) + self.b
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        param = []
        for neuron in self.neurons:
            param.extend(neuron.parameters())
        return param


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        sz_layer = [nin] + nouts
        self.layers = [Layer(sz_layer[i], sz_layer[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        param = []
        for layer in self.layers:
            param.extend(layer.parameters())
        return param

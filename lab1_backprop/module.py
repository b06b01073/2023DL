import numpy as np

class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
        

class Linear:
    def __init__(self, in_features, out_features):
        self.w = np.random.randn(out_features, in_features)
        self.b = np.random.randn(out_features, 1)

    def forward(self, x):
        return np.dot(self.w, x) + self.b


class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
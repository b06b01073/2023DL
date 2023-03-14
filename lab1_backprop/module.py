import numpy as np

class LinearNeuron:
    def __init__(self):
        self.w = np.random.rand()
        self.b = np.random.rand()

    def forward(self, x):
        return self.w * x + self.b

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.neurons = [LinearNeuron() for _ in range(out_features)]

    def forward(self, x):
        assert x.shape[0] == self.in_features, f'input size doesnt fit, the input size of the layer should be {self.in_features}, the given input size is {x.shape[0]}'
        
        y = np.zeros((self.out_features, ))
        for i in range(x.shape[0]):
            for j in range(self.out_features):
                y[j] += self.neurons[j].forward(x[i])

        return y

    def __call__(self, x):
        # send a "batch" of input to the layer, x is of the form (B, n), where B is the batch size, n is the dimension of the input
        assert x.shape[1] == self.in_features, f'input size doesnt fit, the input size of the layer should be {self.in_features}, the given input size is {x.shape[0]}'

        batch_size = x.shape[0]
        y = np.zeros((batch_size, self.out_features))
        for i in range(batch_size):
            y[i] = self.forward(x[i])

        return y


class Sigmoid:
    def __init__(self):
        pass
    
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __call__(self, x):
        print(x.shape)
        batch_size, in_features = x.shape
        y = np.zeros((batch_size, in_features))
        for i in range(batch_size):
            for j in range(in_features):
                y[i][j] = self.forward(x[i][j])

        return y
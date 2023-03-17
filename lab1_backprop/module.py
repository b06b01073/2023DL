import numpy as np

class Linear:
    def __init__(self, in_features, out_features):  
        self.w = np.random.rand(in_features, out_features)
        self.b = np.random.rand(out_features)

    def forward(self, x):
        return self.w.T @ x + self.b

    def __call__(self, x):
        return self.forward(x)


class Sigmoid:

    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    def __call__(self, x):
        return self.forward(x)

def square_error(input1, input2):
    return (input1 - input2) ** 2

def loss_grad(y, x, linear_layer):
    linear_output = linear_layer(x)
    
    sigmoid = Sigmoid()
    sig_output = sigmoid(linear_output)

    grad_w = -(y - sig_output) * sig_output * (1 - sig_output) * np.sum(x)
    grad_b = -(y - sig_output) * sig_output * (1 - sig_output)

    return grad_w, grad_b
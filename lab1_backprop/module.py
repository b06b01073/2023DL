import numpy as np

class Net:
    def __init__(self, layers, lr):
        self.layers = layers
        self.lr = lr

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, pred_grad):
        downstream_grad = pred_grad
        for layer in reversed(self.layers):
            downstream_grad = layer.backward(lr=self.lr, downstream_grad=downstream_grad) if layer.updatable else layer.backward(downstream_grad=downstream_grad)

        

class Linear:
    def __init__(self, in_features, out_features):
        self.w = np.random.randn(out_features, in_features)
        self.b = np.random.randn(out_features, 1)
        self.x = None
        self.updatable = True

    def forward(self, x):
        self.x = x
        return np.dot(self.w, self.x) + self.b

    def backward(self, lr, downstream_grad):
        w_grad = np.dot(downstream_grad, self.x.T)
        b_grad = downstream_grad
        x_grad = np.dot(self.w.T, downstream_grad)


        self.w -= lr * w_grad
        self.b -= lr * b_grad

        return x_grad

class Sigmoid:
    def __init__(self):
        self.x = None
        self.updatable = False

    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-self.x))

    def backward(self, downstream_grad):
        return np.multiply(downstream_grad, self.derivative_sigmoid())

    def derivative_sigmoid(self):
        # note that all the operations are elementwise
        return np.multiply(self.forward(self.x), 1 - self.forward(self.x))

class MSELoss:
    def __init__(self):
        self.pred = None
        self.y = None
        self.size = None

    def forward(self, pred, y):
        # note that the order of pred and y need to be followed strictly, otherwise self.pred and self.y will save the wrong data
        self.pred, self.y, self.size = pred, y, y.size 
        return np.mean((self.pred - self.y) ** 2)

    def backward(self):
        # note that there is no downstream_grad of the loss function, since it is the last node in the computational graph
        return (self.pred - self.y) / self.size
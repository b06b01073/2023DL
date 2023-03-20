import numpy as np

class Net:
    def __init__(self, layers, upper_clip=5, lower_clip=-5):
        self.layers = layers
        self.upper_clip = upper_clip
        self.lower_clip = lower_clip

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, pred_grad, optimizer):
        downstream_grad = pred_grad
        for layer in reversed(self.layers):
            downstream_grad = np.clip(downstream_grad, self.lower_clip, self.upper_clip)
            downstream_grad = layer.backward(optimizer=optimizer, downstream_grad=downstream_grad) if layer.updatable else layer.backward(downstream_grad=downstream_grad)

        

class Linear:
    def __init__(self, in_features, out_features):
        # set w and b in a certain range if overflow 
        self.w = np.random.randn(out_features, in_features) 
        self.b = np.random.randn(out_features, 1)
        self.x = None
        self.updatable = True

    def forward(self, x):
        self.x = x
        return np.dot(self.w, self.x) + self.b

    def backward(self, optimizer, downstream_grad):
        w_grad = np.dot(downstream_grad, self.x.T)
        b_grad = downstream_grad
        x_grad = np.dot(self.w.T, downstream_grad)


        update_w, update_b = optimizer.update(w_grad, b_grad)
        self.w -= update_w
        self.b -= update_b

        return x_grad

class Sigmoid:
    def __init__(self, upper_clip=10, lower_clip=-10):
        self.x = None
        self.updatable = False
        self.upper_clip = upper_clip
        self.lower_clip = lower_clip

    def forward(self, x):
        self.x = np.clip(x, self.lower_clip, self.upper_clip) # the clip function here is to prevent overflow in the np.exp operation
        return 1 / (1 + np.exp(-self.x))

    def backward(self, downstream_grad):
        return np.multiply(downstream_grad, self.derivative_sigmoid())

    def derivative_sigmoid(self):
        # note that all the operations are elementwise
        x = self.forward(self.x)
        return np.multiply(x, 1 - x)
        
class LeakyReLU:
    def __init__(self, upper_clip=10, lower_clip=-10, s=0.2):
        self.x = None
        self.updatable = False
        self.upper_clip = upper_clip
        self.lower_clip = lower_clip
        self.s = s
        self.positive = None
        self.negative = None

    def forward(self, x):
        self.x = np.clip(x, self.lower_clip, self.upper_clip) # the clip function here is to prevent overflow in the np.exp operation
        self.positive = np.where(self.x >= 0)
        self.negative = np.where(self.x < 0)

        output = np.zeros(self.x.shape)
        output[self.positive] = self.x[self.positive]
        output[self.negative] = self.x[self.negative] * self.s

        return output

    def backward(self, downstream_grad):
        return np.multiply(downstream_grad, self.derivative_leaky_relu())

    def derivative_leaky_relu(self):
        derivative = np.zeros(self.x.shape)
        derivative[self.positive] = 1
        derivative[self.negative] = self.s

        return derivative

    
class Tanh:
    def __init__(self, upper_clip=10, lower_clip=-10):
        self.x = None
        self.updatable = False
        self.upper_clip = upper_clip
        self.lower_clip = lower_clip

    def forward(self, x):
        self.x = np.clip(x, self.lower_clip, self.upper_clip) # the clip function here is to prevent overflow in the np.exp operation
        return np.tanh(self.x)

    def backward(self, downstream_grad):
        return np.multiply(downstream_grad, self.derivative_tanh())

    def derivative_tanh(self):
        return 1 / np.cosh(self.x) ** 2



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

class GD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, w_grad, b_grad):
        return w_grad * self.lr, b_grad * self.lr

import numpy as np

class Net_HW1:
    def __init__(self, in_features, out_features):
        self.hidden_features = 128
        self.linear1 = Linear(in_features, self.hidden_features)
        self.linear2 = Linear(self.hidden_features, self.hidden_features)
        self.linear3 = Linear(self.hidden_features, out_features)
        self.sigmoid = Sigmoid()

        self.net = [self.linear1, self.linear2, self.linear3]

    def forward(self, x):
        print(x.shape)
        x = self.sigmoid.forward(self.net[0].forward(x))
        x = self.sigmoid.forward(self.net[1].forward(x))
        x = self.sigmoid.forward(self.net[2].forward(x))

        return x

    # def update(self, loss_func, pred, y):
    #     for i in range(len(self.net) - 1, -1, -1)
    #         layer = self.net[i]

class Linear:
    def __init__(self, in_features, out_features):  

        # the 1e-2 is required, otherwise the sigmoid will always output 1 and break the log function in loss function
        self.w = np.random.rand(out_features, in_features) * 1e-2
        # a dummy dimension is added for broadcasting
        self.b = np.random.rand(out_features, 1)

    def forward(self, x):
        return np.dot(self.w, x) + self.b

    # def backward(self, prev_derivative):
    #     return prev_derivative * 

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    # def backward(self, x):


class LossFunction:
    def forward(self, pred, y):

        # y.shape is (100, 1)
        dataset_size = y.shape[0]
        loss = -(1 / dataset_size) * (np.dot(np.log(pred), y) + np.dot(np.log(1 - pred), (1 - y)))
        return loss.squeeze()

    def backward(self, pred, y):
        return np.divide(1 - y, 1 - pred) - np.divide(y, pred)
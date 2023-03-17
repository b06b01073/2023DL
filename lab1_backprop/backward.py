import numpy as np
import matplotlib.pyplot as plt
import module

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels =  []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()


def main():
    x, y = generate_linear()
    linear1 = module.Linear(in_features=3, out_features=20)
    linear2 = module.Linear(in_features=20, out_features=1)
    sigmoid = module.Sigmoid()

    net = [linear1, sigmoid, linear2, sigmoid]

    epochs = 100
    for i in range(epochs):
        for data, label in zip(x, y):
            pred = data
            for layer in net:
                print(layer)
                pred = layer(pred)
            loss = module.square_error(label, pred[0])
            grad_w, grad_b = module.loss_grad(label, data, linear1)
            linear1.w = linear1.w - 1e-4*grad_w
            linear1.b = linear1.b - 1e-4*grad_b
            print(loss)

    show_result(x, y, pred)

if __name__ == '__main__':
    main()


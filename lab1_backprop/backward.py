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
    layers = [
        module.Linear(in_features=2, out_features=256),
        module.Sigmoid(),
        module.Linear(in_features=256, out_features=64),
        module.Sigmoid(),
        module.Linear(in_features=64, out_features=1)
    ]
    net = module.Net(layers, lr=1e-3)
    criterion = module.MSELoss()

    epochs = 100

    for i in range(epochs):
        total_loss = 0
        for data, label in zip(x, y):
            pred = net.forward(np.expand_dims(data, axis=0).T)
            total_loss += criterion.forward(pred, label)
            pred_grad = criterion.backward() 
            net.backward(pred_grad)
    
    final_pred = []
    for data in x:
        final_pred.append(1 if net.forward(np.expand_dims(data, axis=0).T) >= 0.5 else 0)

    show_result(x, y, final_pred)

if __name__ == '__main__':
    main()


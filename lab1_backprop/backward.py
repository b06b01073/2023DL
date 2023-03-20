import numpy as np
import matplotlib.pyplot as plt
import module
import argparse

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
    plt.savefig('result')
    plt.show()

def threshold(x):
    return 1 if x >= 0.5 else 0

def get_activation(activation):
    if activation == 'sigmoid':
        print('Sigmoid')
        return module.Sigmoid
    elif activation == 'tanh':
        print('Tanh')
        return module.Tanh
    elif activation == 'leakyrelu':
        print('relu')
        return module.LeakyReLU

def main(args):
    task = generate_linear if args.task == 'linear' else generate_XOR_easy
    activation = get_activation(args.activation)
    x, y = task()

    # note that sometimes it might perform horribly on XOR task if the number of neurons are not large enough 
    hidden_layers_features = [256, 64, 32, 16]
    layers = [
        module.Linear(in_features=2, out_features=hidden_layers_features[0]),
        activation(),
        module.Linear(in_features=hidden_layers_features[0], out_features=hidden_layers_features[1]),
        activation(),
        module.Linear(in_features=hidden_layers_features[1], out_features=1),
    ]
    net = module.Net(layers, lr=args.lr)
    criterion = module.MSELoss()

    epochs = 1000
    eps = 1e-3
    prev_total_loss = 0
    loss_history = []


    for i in range(epochs):
        total_loss = 0
        for data, label in zip(x, y):
            pred = net.forward(np.expand_dims(data, axis=0).T)
            total_loss += criterion.forward(pred, label)
            pred_grad = criterion.backward() 
            net.backward(pred_grad)
        print(f'epoch {i} loss : {total_loss}')
        loss_history.append(total_loss)
        if np.abs(prev_total_loss - total_loss) < eps:
            break
    
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(epochs), loss_history)
    plt.savefig('epoch_loss')
    plt.clf()

    test(net, x, y, criterion)

def test(net, x, y, criterion):
    final_pred = []
    correct = 0
    total_loss = 0
    for idx, (data, label) in enumerate(zip(x, y)):
        output = net.forward(np.expand_dims(data, axis=0).T)
        total_loss += criterion.forward(output, label)
        label = label.squeeze()


        pred = threshold(output)
        final_pred.append(pred)

        if pred == label:
            correct += 1

        print(f'Iter{idx} | \t Ground truth: {label.squeeze()}| \t prediction: {output} |')

    print(f'loss={total_loss} accuracy={correct / y.size * 100}%')

    
    show_result(x, y, final_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default='linear')
    parser.add_argument('--lr', '-l', type=float, default=1e-2)
    parser.add_argument('--activation', '-a', type=str, default='sigmoid')

    args = parser.parse_args()
    main(args)


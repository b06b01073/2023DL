import argparse
from dataloader import read_bci_data
from model import EEGNet
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

def augmentation(data_batches):
    augmented_batches = deepcopy(data_batches)
    for i in range(len(augmented_batches)):
        # print(augmented_batches[i].shape)
        if np.random.uniform() > 0.6:
            augmented_batches[i] += np.clip(np.random.normal(scale=5e-3, size=augmented_batches[i].shape), -1, 1)
        # if np.random.uniform() > 0.8:
        #     augmented_batches[i] = np.flip(augmented_batches[i], axis=2)

    return augmented_batches

def get_batches(data, label, batch_size):
    dataset_size = data.shape[0]
    batches = dataset_size // batch_size
    remain = (dataset_size % batch_size == 0)

    data_batches = []
    label_batches = []

    for batch in range(batches):
        if remain and batch == batches - 1:
            mini_batch = data[batch*batch_size:]
            labels = label[batch*batch_size:]
        else:
            mini_batch = data[batch*batch_size: (batch+1)*batch_size]
            labels = label[batch*batch_size: (batch+1)*batch_size]

        data_batches.append(mini_batch)
        label_batches.append(labels)

    return data_batches, label_batches

def train(data_batches, label_batches, model, optimizer, loss_fn, dataset_size):
    model.train()
    total_loss = 0
    corrects = 0

    data_batches = augmentation(data_batches)

    for mini_batch, labels in zip(data_batches, label_batches):
        mini_batch = torch.from_numpy(mini_batch).float().to(device)
        labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
        output = model(mini_batch)

        optimizer.zero_grad()
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss

        with torch.no_grad():
            preds = torch.argmax(output, dim=1)
            corrects += torch.sum(preds == labels).item()

    return total_loss.item(), corrects / dataset_size

def test(data_batches, label_batches, model, loss_fn, dataset_size):
    model.eval()
    total_loss = 0
    corrects = 0
    with torch.no_grad():
        for mini_batch, labels in zip(data_batches, label_batches):
            mini_batch = torch.from_numpy(mini_batch).float().to(device)
            labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
            output = model(mini_batch)

            total_loss += loss_fn(output, labels)

            preds = torch.argmax(output, dim=1)
            corrects += torch.sum(preds == labels).item()

        return total_loss.item(), corrects / dataset_size

def main(args):
    train_data, train_label, test_data, test_label = read_bci_data()
    train_data_batches, train_label_batches = get_batches(train_data, train_label, args.batch_size)
    test_data_batches, test_label_batches = get_batches(test_data, test_label, args.batch_size)

    epoch = args.epoch

    print(f'Train data shape: {train_data.shape}, Train label shape: {train_label.shape}')
    print(f'Test data shape: {test_data.shape}, Test label shape: {test_label.shape}')

    model = EEGNet(args.activation).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

    train_size = train_data.shape[0]
    test_size = test_data.shape[0]

    for i in range(epoch):
        train_loss, train_acc = train(train_data_batches, train_label_batches, model, optimizer, loss_fn, train_size)
        test_loss, test_acc = test(test_data_batches, test_label_batches, model, loss_fn, test_size)
        scheduler.step()

        print(f'Epoch: {i}')
        print(f'train loss: {train_loss}, train_acc: {train_acc}')
        print(f'test loss: {test_loss}, test_acc: {test_acc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', '-e', type=int, default=300)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--activation', '-a', type=str, default='relu')

    args = parser.parse_args()
    main(args)
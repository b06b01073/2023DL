import argparse
from dataloader import read_bci_data
from model import EEGNet
import torch
import torch.optim as optim
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

def train(train_data, train_label, batch_size, model, optimizer, loss_fn):
    dataset_size = train_data.shape[0]
    batches = dataset_size // batch_size
    remain = (dataset_size % batch_size == 0)

    if remain:
        batches += 1

    total_loss = 0

    for batch in range(batches):
        if remain and batch == batches - 1:
            mini_batch = train_data[batch*batch_size:]
            labels = train_label[batch*batch_size:]
        else:
            mini_batch = train_data[batch*batch_size: (batch+1)*batch_size]
            labels = train_label[batch*batch_size: (batch+1)*batch_size]

        mini_batch = torch.from_numpy(mini_batch).float().to(device)
        labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)
        preds = model(mini_batch)


        optimizer.zero_grad()
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss

    print(total_loss.item())




def test(test_data, test_label):
    pass

def main(args):
    train_data, train_label, test_data, test_label = read_bci_data()
    epoch = args.epoch

    print(f'Train data shape: {train_data.shape}, Train label shape: {train_label.shape}')
    print(f'Test data shape: {test_data.shape}, Test label shape: {test_label.shape}')

    model = EEGNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for i in range(epoch):
        train(train_data, train_label, args.batch_size, model, optimizer, loss_fn)
        test(test_data, test_label)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', '-e', type=int, default=150)
    parser.add_argument('--batch_size', '-b', type=int, default=64)

    args = parser.parse_args()
    main(args)
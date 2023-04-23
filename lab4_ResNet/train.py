from argparse import ArgumentParser
import utils
import dataloader
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torchsampler import ImbalancedDatasetSampler



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')

def train(args):
    # class_counts = [20655, 1955, 4210, 698, 5811]
    # sample_weights = np.array([sum(class_counts) / class_counts[i] for i in range(len(class_counts))])

    # sample_weights = torch.from_numpy(sample_weights)
    # sample_weights = sample_weights.double()
    # print(sample_weights)


    model = utils.get_model(args.model, pretrained=False).to(device)
    print(model)
    train_dataset = dataloader.RetinopathyLoader(args.train_root, mode='train')

    # sampler = WeightedRandomSampler(weights=sample_weights, num_samples=2000, replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=12)
    
    test_dataset = dataloader.RetinopathyLoader(args.test_root, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=12)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_correct = 0

    for i in range(args.epoch):
        model.train()
        train_loss = 0
        test_loss = 0
        train_correct = 0
        test_correct = 0
        for imgs, labels in tqdm(train_loader, desc=f'epoch: {i} (train)'):
            imgs, labels = imgs.to(device), labels.to(device)

            preds = model(imgs)

            optimizer.zero_grad()
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()


            with torch.no_grad():
                train_loss += loss.item()
                train_correct += torch.sum(torch.argmax(preds, dim=1) == labels)

        print(f'train_loss: {train_loss}, train_acc: {train_correct / len(train_dataset)}')
        with torch.no_grad():
            test_res = test(model, test_loader, loss_fn, i)
            test_correct = test_res[1]
            test_loss += test_res[0]
            if test_correct >= best_correct:
                best_correct = test_correct
                print('saved')
                torch.save(model.state_dict(), f'{args.model}.pth')

        print(f'test_loss: {test_loss}, test_acc: {test_correct / len(test_dataset)}')


def test(model, test_loader, loss_fn, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    for imgs, labels in tqdm(test_loader, desc=f'epoch: {epoch} (test)'):
        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs)
        loss = loss_fn(preds, labels)
        total_loss += loss.item()
        correct += torch.sum(labels == torch.argmax(preds, dim=1))
        
    return total_loss, correct


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='resnet18')
    parser.add_argument('--train_root', type=str, default='./dataset/new_train')
    parser.add_argument('--test_root', type=str, default='./dataset/new_test')
    parser.add_argument('--batch_size', '-b', type=int, default=4)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--epoch', '-e', type=int, default=10)

    args = parser.parse_args()

    train(args)
from argparse import ArgumentParser
import utils
import dataloader
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device: {device}')

def train(args):
    model = utils.get_model(args.model, pretrained=True).to(device)
    train_dataset = dataloader.RetinopathyLoader(args.train_root, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dataset = dataloader.RetinopathyLoader(args.test_root, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0

    for i in range(args.epoch):
        model.train()
        train_loss = 0
        test_loss = 0
        train_correct = 0
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

        with torch.no_grad():
            test_res = test(model, test_loader, loss_fn, i)
            test_acc = test_res[1]
            test_loss += test_res[0]
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f'{args.model}.pth')

        print(f'train_loss: {train_loss}, train_acc: {train_acc / len(train_loader)}')
        print(f'test_loss: {test_loss}, test_acc: {test_acc}')


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
        
    return total_loss, correct / len(test_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='resnet18')
    parser.add_argument('--train_root', type=str, default='./dataset/new_train')
    parser.add_argument('--test_root', type=str, default='./dataset/new_test')
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    parser.add_argument('--lr', '-l', type=float, default=1e-3)
    parser.add_argument('--epoch', '-e', type=int, default=10)

    args = parser.parse_args()

    train(args)
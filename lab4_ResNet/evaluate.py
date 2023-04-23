import matplotlib.pyplot as plt
import torch
from argparse import ArgumentParser
import utils
from tqdm import tqdm
from torch.utils.data import DataLoader
import dataloader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device}')


def evaluate(args):
    model = utils.get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.path))
    model.eval()

    test_dataset = dataloader.RetinopathyLoader(args.test_root, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=12)

    corrects = 0
    y_pred = []
    y_true = []

    for imgs, labels in tqdm(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        preds = model(imgs)
        corrects += torch.sum(labels == torch.argmax(preds, dim=1))
        
        y_true += list(labels.to('cpu'))
        y_pred += list(torch.argmax(preds, dim=1).to('cpu'))

    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, normalize='true', display_labels=[i for i in range(5)])
    disp.plot()
    plt.savefig('result.jpg')
    print(corrects.item() / len(test_dataset))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='resnet18')
    parser.add_argument('--path', '-p', type=str, default='pretrained/resnet18.pth')
    parser.add_argument('--batch_size', '-b', type=int, default=8)
    parser.add_argument('--test_root', type=str, default='./dataset/new_test')

    args = parser.parse_args()

    with torch.no_grad():
        evaluate(args)
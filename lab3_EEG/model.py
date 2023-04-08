import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EEGNet(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.activation = get_activation(activation)
        print(f'Initializing model {self}...')
        print(f'Using activation function {self.activation}')

        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(num_features=16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            self.activation,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

        self.classify = nn.Sequential(
            nn.Linear(736, 2, bias=True),
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = torch.flatten(x, 1)
        x = self.classify(x)

        return x


class DeepConvNet(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()

        self.activation = get_activation(activation)
        print(f'Initializing model {self}...')
        print(f'Using activation function {self.activation}')

        self.header = nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5))

        self.hidden_dims = [25, 25, 50, 100, 200]
        self.hidden_kernel_size = [(2, 1), (1, 5), (1, 5), (1, 5)]
        self.pool_kernel_size = (1, 2)

        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_dims[0], out_channels=self.hidden_dims[1], kernel_size=self.hidden_kernel_size[0]),
            nn.BatchNorm2d(self.hidden_dims[1]),
            self.activation,
            nn.MaxPool2d(kernel_size=self.pool_kernel_size),
            nn.Dropout(p=0.5),
        )
        self.conv_layers2 = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_dims[1], out_channels=self.hidden_dims[2], kernel_size=self.hidden_kernel_size[1]),
            nn.BatchNorm2d(self.hidden_dims[2]),
            self.activation,
            nn.MaxPool2d(kernel_size=self.pool_kernel_size),
            nn.Dropout(p=0.5),
        )
        self.conv_layers3 = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_dims[2], out_channels=self.hidden_dims[3], kernel_size=self.hidden_kernel_size[2]),
            nn.BatchNorm2d(self.hidden_dims[3]),
            self.activation,
            nn.MaxPool2d(kernel_size=self.pool_kernel_size),
            nn.Dropout(p=0.5),
        )
        self.conv_layers4 = nn.Sequential(
            nn.Conv2d(in_channels=self.hidden_dims[3], out_channels=self.hidden_dims[4], kernel_size=self.hidden_kernel_size[3]),
            nn.BatchNorm2d(self.hidden_dims[4]),
            self.activation,
            nn.MaxPool2d(kernel_size=self.pool_kernel_size),
            nn.Dropout(p=0.5),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=8600, out_features=2),
        )
    
    def forward(self, x):
        x = self.header(x)
        x = self.conv_layers1(x)
        x = self.conv_layers2(x)
        x = self.conv_layers3(x)
        x = self.conv_layers4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ShallowConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        print(f'Initializing {self}...')

        self.header = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 13))

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(2, 1)),
            nn.BatchNorm2d(40),
        )

        self.pool = nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7))

        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(in_features=4040, out_features=2)
    
    def forward(self, x):
        x = self.header(x)
        x = self.conv_layer(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(x)
        x = self.drop(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def get_model(model, activation):
    if model == 'eegnet':
        return EEGNet(activation)
    elif model == 'deep':
        return DeepConvNet(activation)
    elif model == 'shallow':
        return ShallowConvNet()

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU()
    elif activation == 'elu':
        return nn.ELU(alpha=1.0)
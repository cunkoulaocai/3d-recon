import torch as tr
from torch import nn
from torch.nn import functional as F

from config import Config
from modules import Conv2DBlock, DenseBlock, ConvT3DBlock


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.conv1 = Conv2DBlock(1, 128, batch_norm=config.e_use_bn, layer_norm=False,
                                 kernel_size=5, stride=2, padding='same', act='relu')
        self.conv2 = Conv2DBlock(128, 256, batch_norm=config.e_use_bn, layer_norm=False,
                                 kernel_size=5, stride=2, padding='same', act='relu')
        self.conv3 = Conv2DBlock(256, 512, batch_norm=config.e_use_bn, layer_norm=False,
                                 kernel_size=5, stride=2, padding='same', act='relu')

        self.fc1 = DenseBlock(4 * 4 * 512, 1024, batch_norm=config.e_use_bn, act='relu')

        self.fc2 = DenseBlock(1024, 1024, batch_norm=config.e_use_bn, act='relu')

        self.fc3 = DenseBlock(1024, config.z_dim, batch_norm=False, act=None)

        self.cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1, 4 * 4 * 512)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class Decoder(nn.Module):

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config

        self.fc1 = DenseBlock(config.z_dim, 256 * 4 * 4 * 4, batch_norm=True, act='relu')

        self.tconv1 = ConvT3DBlock(256, 128, batch_norm=self.config.g_use_bn, layer_norm=False, kernel_size=5, stride=2,
                                   padding=(2, 2, 2), output_padding=1, act='relu')

        self.tconv2 = ConvT3DBlock(128, 64, batch_norm=self.config.g_use_bn, layer_norm=False, kernel_size=5, stride=2,
                                   padding=(2, 2, 2), output_padding=1, act='relu')

        self.tconv3 = ConvT3DBlock(64, 1, batch_norm=False, layer_norm=False, kernel_size=5, stride=2,
                                   padding=(2, 2, 2), output_padding=1, act='sigmoid')

        self.cuda()

    def forward(self, x):
        x = self.fc1(x)

        x = x.view(-1, 256, 4, 4, 4)

        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)

        return x


class Disc(nn.Module):
    def __init__(self, config):
        super(Disc, self).__init__()
        self.config = config
        self.conv1 = Conv2DBlock(1, 256, batch_norm=config.d_use_bn, layer_norm=config.d_use_ln,
                                 kernel_size=5, stride=2, padding='same', act='lrelu', ln_shape=(256, 16, 16))
        self.conv2 = Conv2DBlock(256, 512, batch_norm=config.d_use_bn, layer_norm=config.d_use_ln,
                                 kernel_size=5, stride=2, padding='same', act='lrelu', ln_shape=(512, 8, 8))
        self.conv3 = Conv2DBlock(512, 1024, batch_norm=config.d_use_bn, layer_norm=config.d_use_ln,
                                 kernel_size=5, stride=2, padding='same', act='lrelu', ln_shape=(1024, 4, 4))

        self.fc1 = nn.Linear(1024 * 4 * 4, 1)

        self.cuda()

    def forward(self, x):
        '''
        Returns the logits before the sigmoid activation
        :param x: binary image
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1, 1024 * 4 * 4)

        x = self.fc1(x)

        return x

    def predict_probs(self, x):
        logits = self.forward(x)
        probs = tr.sigmoid(logits)
        return probs

    def predict(self, x):
        logits = self.forward(x)
        return (logits >= 0).float()

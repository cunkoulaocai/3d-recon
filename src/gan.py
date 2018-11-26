import torch as tr
from torch import nn, optim
from torch.nn import functional as F

from nets import Decoder
from nets import Disc
from nets import Encoder
from projector import Projector


class GAN(nn.Module):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.config = config
        self.e_net = Encoder(config)
        self.g_net = Decoder(config)
        self.d_net = Disc(config)
        self.p_net = Projector(config)

        self._build_optimizers()

    def _build_optimizers(self):
        weight_decay = 1e-4
        if self.config.optimizer_type.lower() == 'rmsprop':
            self.e_opt = optim.RMSprop(self.e_net.parameters(), weight_decay=weight_decay)
            self.g_opt = optim.RMSprop(self.g_net.parameters(), weight_decay=weight_decay)
            self.g_adv_opt = optim.RMSprop(self.g_net.parameters(), weight_decay=weight_decay)
            self.d_adv_opt = optim.RMSprop(self.d_net.parameters(), weight_decay=weight_decay)

        elif self.config.optimizer_type.lower() == 'adam':
            self.e_opt = optim.Adam(self.e_net.parameters(), weight_decay=weight_decay)
            self.g_opt = optim.Adam(self.g_net.parameters(), weight_decay=weight_decay)
            self.g_adv_opt = optim.Adam(self.g_net.parameters(), weight_decay=weight_decay)
            self.d_adv_opt = optim.Adam(self.d_net.parameters(), weight_decay=weight_decay)

    def compute_ae_losses(self, x1, p1, x2, p2):
        """
        Compute Auto Encoder Losses: L_reconstruction, L_pose_invariance (on representations), L_pose_invariance(on voxels)
        :param x1: Image 1
        :param p1: Camera Angles for Image 1
        :param x2: Image 2
        :param p2: Camera Angles for Image 2
        :return: l_autoencoder, l_recon, l_pinv, l_vinv
        """
        z1 = self.e_net(x1)
        z2 = self.e_net(x2)

        v1 = self.g_net(z1)
        v2 = self.g_net(z2)

        x_1 = self.p_net(v2, p1)
        x_2 = self.p_net(v1, p2)

        l1_loss = F.l1_loss(x1, x_1) + F.l1_loss(x2, x_2)
        l2_loss = F.mse_loss(x1, x_1) + F.mse_loss(x2, x_2)
        l_recon = l1_loss + l2_loss

        l_pinv = F.mse_loss(z1, z2)

        l_vinv = F.mse_loss(v1, v2)

        l_autoencoder = (
                self.config.content_loss_weight * l_recon
                + self.config.pose_inv_loss_weight * l_pinv
                + self.config.vox_inv_loss_weight * l_vinv
        )
        return l_autoencoder, l_recon, l_pinv, l_vinv

    def compute_adv_losses(self, x, z, p, report_accurcies=True):
        v_fake = self.g_net(z)
        x_fake = self.p_net(v_fake, p)

        r_logits = self.d_net(x)
        f_logits = self.d_net(x_fake)

        if self.config.gan_loss_type.lower() == 'dcgan':
            # DCGAN Implementation for Adversarial Losses
            d_real_loss = F.binary_cross_entropy_with_logits(r_logits, tr.ones_like(r_logits))
            d_fake_loss = F.binary_cross_entropy_with_logits(f_logits, tr.zeros_like(f_logits))
            g_loss = F.binary_cross_entropy_with_logits(f_logits, tr.ones_like(r_logits))

        elif self.config.gan_loss_type.lower() == 'lsgsn':
            # LSGAN Implementation for Adversarial Losses
            d_real_loss = F.mse_loss(r_logits, tr.ones_like(r_logits))
            d_fake_loss = F.mse_loss(f_logits, -tr.ones_like(f_logits))
            g_loss = F.mse_loss(f_logits, tr.ones_like(r_logits))

        d_loss = 0.5 * (d_real_loss + d_fake_loss)

        if report_accurcies:
            # Predictions for real samples
            r_preds = (r_logits >= 0).float()

            # Predictions for generated samples
            f_preds = (f_logits >= 0).float()

            d_real_acc = tr.mean(r_preds) * 100
            d_fake_acc = tr.mean(1 - f_preds) * 100

            d_acc = 0.5 * (d_real_acc + d_fake_acc)

            g_acc = tr.mean(f_preds) * 100

            return d_loss, g_loss, d_acc, g_acc
        else:
            return d_loss, g_loss

    def compute_gan_accuracies(self, x, z, p):
        """ Discriminator and Generator Accuracies"""
        v_fake = self.g_net(z)
        x_fake = self.p_net(v_fake, p)

        r_preds = self.d_net.predict(x)
        f_preds = self.d_net.predict(x_fake)

        d_real_acc = tr.mean(r_preds) * 100
        d_fake_acc = tr.mean(1 - f_preds) * 100

        d_acc = 0.5 * (d_real_acc + d_fake_acc)

        g_acc = tr.mean(f_preds) * 100

        return d_acc, g_acc

    def step_train_autoencoder(self, x1, p1, x2, p2):
        """ Single Step Encoder Decoder training using Consistency Losses"""
        l_autoencoder, l_recon, l_pinv, l_vinv = self.compute_ae_losses(x1, p1, x2, p2)
        self.e_opt.zero_grad()
        self.g_opt.zero_grad()

        l_autoencoder.backward()

        self.e_opt.step()
        self.g_opt.step()

        return l_autoencoder, l_recon, l_pinv, l_vinv

    def step_train_gan(self, x, z, p, mode, report_accuracies=True):
        """ Single Step Decoder | Generator training using Adversarial Losses"""
        measures = self.compute_adv_losses(x, z, p, report_accuracies)

        d_loss, g_loss = measures[:2]

        if mode.lower() == 'g':
            self.d_adv_opt.zero_grad()
            d_loss.backward()
            self.d_adv_opt.step()
        else:
            self.g_adv_opt.zero_grad()
            g_loss.backward()
            self.g_adv_opt.step()

        return measures

    def predict_voxel(self, img):
        """ Predict Voxel given a single view projection image"""
        with tr.no_grad():
            vox = self.g_net(self.e_net(img))
            return vox

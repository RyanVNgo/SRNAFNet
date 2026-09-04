
import torch
import torch.nn as nn

from torch.amp import GradScaler, autocast


class Discriminator(nn.Module):
    def __init__(self, iterations, device='cpu'):
        super().__init__()
        self.net = d_net().to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=iterations,
            eta_min=1e-7
        )
        self.scaler = GradScaler()


    def forward(self, x):
        with autocast(self.device):
            logits = self.net(x)
            return logits

    def adversarial_loss(self, pred, target):
        with autocast(self.device):
            real_p = self.net(target.detach().to(self.device))
            fake_p = self.net(pred.to(self.device))

        dRF = torch.sigmoid(real_p - torch.mean(fake_p))
        dFR = torch.sigmoid(fake_p - torch.mean(real_p))
        eR = torch.mean(torch.log(1 - dRF))
        eF = torch.mean(torch.log(dFR))
        g_loss = -eR - eF
        return g_loss

    def update(self, pred, target):
        with autocast(self.device):
            real_p = self.net(target.to(self.device))
            fake_p = self.net(pred.detach().to(self.device))

        dRF = torch.sigmoid(real_p - torch.mean(fake_p))
        dFR = torch.sigmoid(fake_p - torch.mean(real_p))
        eR = torch.mean(torch.log(dRF))
        eF = torch.mean(torch.log(1 - dFR))
        d_loss = -eR - eF
        self.optimizer.zero_grad()
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return d_loss


class d_net(nn.Module):
    def __init__(self):
        super().__init__()
        chan = 64

        self.intro = nn.Sequential(
            nn.Conv2d(3, chan, 3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(chan, chan, 3, stride=2),
            nn.BatchNorm2d(chan)
        )

        self.body = nn.ModuleList()
        for _ in range(5):
            self.body.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan*2, 3, stride=1, padding='same'),
                    nn.BatchNorm2d(chan*2),
                    nn.LeakyReLU()
                )
            )
            chan = chan * 2
            self.body.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan, 3, stride=2),
                    nn.BatchNorm2d(chan),
                    nn.LeakyReLU()
                )
            )

        self.ending = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(chan, 1024, 1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1, 1),
        )

    def forward(self, x):
        x = self.intro(x)
        for block in self.body:
            x = block(x)
        x = self.ending(x)
        # x = nn.Sigmoid()(x)
        return x


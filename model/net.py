import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from model.basenet import BaseNet
from model.loss import VGGLoss
from model.layer import init_weights, ConfidenceDrivenMaskLayer
import numpy as np
from util.utils import generate_mask
from functools import reduce
from torch.optim import lr_scheduler


# generative multi-column convolutional neural net
class DMFB(nn.Module):
    def __init__(self):
        super(DMFB, self).__init__()
        self.conv_3 = nn.Conv2d(256, 64, 3, 1, 1)
        conv_3_sets = []
        for i in range(4):
            conv_3_sets.append(nn.Conv2d(64, 64, 3, padding=1))
        self.conv_3_sets = nn.ModuleList(conv_3_sets)
        self.conv_3_2 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.conv_3_4 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)
        self.conv_3_8 = nn.Conv2d(64, 64, 3, padding=8, dilation=8)
        self.act_fn = nn.Sequential(nn.ReLU(), nn.InstanceNorm2d(256))
        self.conv_1 = nn.Conv2d(256, 256, 1)
        self.norm = nn.InstanceNorm2d(256)

    def forward(self, inputs):
        src = inputs
        # conv-3
        x = self.act_fn(inputs)
        x = self.conv_3(x)
        K = []
        for i in range(4):
            if i != 0:
                p = eval('self.conv_3_' + str(2 ** i))(x)
                p = p + K[i - 1]
            else:
                p = x
            K.append(self.conv_3_sets[i](p))
        cat = torch.cat(K, 1)
        bottle = self.conv_1(self.norm(cat))
        out = bottle + src
        return out


class DFBN(BaseNet):
    def __init__(self):
        super(DFBN, self).__init__()
        # 不需要激活函数？
        self.basemodel = nn.Sequential(
            nn.Conv2d(4, 64, 5, 1, 2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.InstanceNorm2d(256),
            nn.Sequential(*[DMFB() for _ in range(24)]),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
            )

    def forward(self, inputs):
        output = self.basemodel(inputs)
        return output


# return one dimensional output indicating the probability of realness or fakeness
class Discriminator(BaseNet):
    def __init__(self, in_channels, cnum=64, is_global=True, act=F.leaky_relu):
        super(Discriminator, self).__init__()
        self.act = act
        self.embedding = None
        self.logit = None

        ch = cnum
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels, ch, kernel_size=5, padding=2, stride=2))
        self.layers.append(nn.BatchNorm2d(ch))
        self.layers.append(nn.Conv2d(ch, ch * 2, kernel_size=5, padding=2, stride=2))
        self.layers.append(nn.BatchNorm2d(ch * 2))
        self.layers.append(nn.Conv2d(ch * 2, ch * 4, kernel_size=5, padding=2, stride=2))
        self.layers.append(nn.BatchNorm2d(ch * 4))
        self.layers.append(nn.Conv2d(ch * 4, ch * 8, kernel_size=5, padding=2, stride=2))
        self.layers.append(nn.BatchNorm2d(ch * 8))
        self.layers.append(nn.Conv2d(ch * 8, ch * 8, kernel_size=5, padding=2, stride=2))
        self.layers.append(nn.BatchNorm2d(ch * 8))
        if is_global:
            self.layers.append(nn.Conv2d(ch * 8, ch * 8, kernel_size=5, padding=2, stride=2))
            self.layers.append(nn.BatchNorm2d(ch * 8))
            self.is_global = True
        else:
            self.is_global = False
        self.layers.append(nn.Linear(ch * 8 * 4 * 4, 512))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, middle_output=False):
        bottleneck = []
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = self.act(x)
                bottleneck += [x]
        if self.is_global:
            bottleneck = bottleneck[:-1]
        self.embedding = x.view(x.size(0), -1)
        self.logit = self.layers[-1](self.embedding)
        if middle_output:
            return bottleneck
        else:
            return self.logit


class GlobalLocalDiscriminator(BaseNet):
    def __init__(self, in_channels, cnum=32, act=F.leaky_relu):
        super(GlobalLocalDiscriminator, self).__init__()
        self.act = act

        self.global_discriminator = Discriminator(in_channels=in_channels, is_global=True, cnum=cnum,
                                                  act=act)
        self.local_discriminator = Discriminator(in_channels=in_channels, is_global=False, cnum=cnum,
                                                 act=act)
        self.liner = nn.Linear(1024, 1)
        self.l1 = nn.L1Loss()

    def forward(self, mode, *input):
        if mode == 'dis':
            return self.forward_adv(*input)
        elif mode == 'adv':
            return self.forward_adv(*input)
        else:
            return self.forward_fm_dis(*input)

    def forward_adv(self, x_g, x_l):
        x_global = self.global_discriminator(x_g)
        x_local = self.local_discriminator(x_l)
        ca = torch.cat([x_global, x_local], -1)
        logit = self.liner(F.leaky_relu(ca))
        return logit

    def forward_fm_dis(self, real, fake, weight_fn):
        Dreal = self.local_discriminator(real, middle_output=True)
        Dfake = self.local_discriminator(fake, middle_output=True)
        fm_dis_list = []
        for i in range(5):
            fm_dis_list += [F.l1_loss(Dreal[i], Dfake[i], reduction='sum') * weight_fn(Dreal[i])]
        fm_dis = reduce(lambda x, y: x + y, fm_dis_list)
        return fm_dis


class InpaintingModel_DFBM(BaseModel):
    def __init__(self, act=F.elu, opt=None):
        super(InpaintingModel_DFBM, self).__init__()
        self.opt = opt
        self.init(opt)

        self.confidence_mask_layer = ConfidenceDrivenMaskLayer()

        self.netDFBN = DFBN().cuda()
        init_weights(self.netDFBN)
        self.model_names = ['DFBN']
        if self.opt.phase == 'test':
            return

        self.netD = None
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netDFBN.parameters(), lr=opt.lr, betas=(0.5, 0.9))
        self.optimizers += [self.optimizer_G]
        self.optimizer_D = None
        self.zeros = torch.zeros((opt.batch_size, 1)).cuda()
        self.ones = torch.ones((opt.batch_size, 1)).cuda()
        self.aeloss = nn.L1Loss()
        self.vggloss = None
        self.G_loss = None
        self.G_loss_mrf = None
        self.G_loss_adv, self.G_loss_vgg, self.G_loss_fm_dis = None, None, None
        self.G_loss_ae = None
        self.loss_eta = 5
        self.loss_mu = 0.03
        self.loss_vgg = 1
        self.BCEloss = nn.BCEWithLogitsLoss().cuda()
        self.gt, self.gt_local = None, None
        self.mask, self.mask_01 = None, None
        self.rect = None
        self.im_in, self.gin = None, None

        self.completed, self.completed_local = None, None
        self.completed_logit, self.gt_logit = None, None

        def weight_fn(layer):
            s = layer.shape
            return 1e3 / (s[1] * s[1] * s[1] * s[2] * s[3])

        self.weight_fn = weight_fn

        self.pred = None

        self.netD = GlobalLocalDiscriminator(3, cnum=opt.d_cnum, act=F.leaky_relu).cuda()
        init_weights(self.netD)
        self.optimizer_D = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netD.parameters()), lr=opt.lr,
                                            betas=(0.5, 0.9))
        self.vggloss = VGGLoss()
        self.optimizers += [self.optimizer_D]
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, [2000, 40000], 0.5))

    def get_current_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def update_learning_rate(self):
        for schedular in self.schedulers:
            schedular.step()

    def initVariables(self):
        self.gt = self.input['gt']
        mask, rect = generate_mask(self.opt.mask_type, self.opt.img_shapes, self.opt.mask_shapes)
        self.mask_01 = torch.from_numpy(mask).cuda().repeat([self.opt.batch_size, 1, 1, 1])
        self.mask = self.confidence_mask_layer(self.mask_01)
        if self.opt.mask_type == 'rect':
            self.rect = [rect[0, 0], rect[0, 1], rect[0, 2], rect[0, 3]]
            self.gt_local = self.gt[:, :, self.rect[0]:self.rect[0] + self.rect[1],
                            self.rect[2]:self.rect[2] + self.rect[3]]
        else:
            self.gt_local = self.gt
        self.im_in = self.gt * (1 - self.mask_01)
        self.gin = torch.cat((self.im_in, self.mask_01), 1)

    def Dra(self, x1, x2):
        return x1 - torch.mean(x2)

    def forward_G(self):
        # self.G_loss_reconstruction = self.recloss(self.completed * self.mask, self.gt.detach() * self.mask)
        # self.G_loss_reconstruction = self.G_loss_reconstruction / torch.mean(self.mask_01)
        self.G_loss_ae = self.aeloss(self.completed_local, self.gt_local)

        # vgg loss
        mask_error = torch.mean(F.mse_loss(self.completed_local, self.gt_local, reduction='none'), dim=1)
        mask_max = mask_error.max(1, True)[0].max(2, True)[0]
        mask_min = mask_error.min(1, True)[0].min(2, True)[0]
        mask_guidance = (mask_error - mask_min) / (mask_max - mask_min)
        self.G_loss_vgg = self.vggloss(self.completed_local, self.gt_local.detach(), mask_guidance.detach(), self.weight_fn)

        # adv loss
        xf = self.netD('adv', self.completed, self.completed_local)
        xr = self.netD('adv', self.gt, self.gt_local)
        self.G_loss_adv = (self.BCEloss(self.Dra(xr, xf), self.zeros) + self.BCEloss(self.Dra(xf, xr), self.ones)) / 2

        # fm dis loss
        self.G_loss_fm_dis = self.netD('fm_dis', self.gt_local, self.completed_local, self.weight_fn)

        self.G_loss = self.G_loss_ae + self.loss_vgg * self.G_loss_vgg + self.loss_mu * self.G_loss_adv + self.loss_eta * self.G_loss_fm_dis

    def forward_D(self):
        xf = self.netD('dis', self.completed.detach(), self.completed_local.detach())
        xr = self.netD('dis', self.gt, self.gt_local)
        # hinge loss
        self.D_loss = (self.BCEloss(self.Dra(xr, xf), self.ones) + self.BCEloss(self.Dra(xf, xr), self.zeros)) / 2

    def backward_G(self):
        self.G_loss.backward()

    def backward_D(self):
        self.D_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        self.initVariables()

        self.pred = self.netDFBN(self.gin)
        self.completed = self.pred * self.mask_01 + self.gt * (1 - self.mask_01)
        if self.opt.mask_type == 'rect':
            self.completed_local = self.completed[:, :, self.rect[0]:self.rect[0] + self.rect[1],
                                   self.rect[2]:self.rect[2] + self.rect[3]]
        else:
            self.completed_local = self.completed

        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.forward_G()
        self.backward_G()
        self.optimizer_G.step()

        for p in self.netD.parameters():
            p.requires_grad = True

        for i in range(self.opt.D_max_iters):
            self.optimizer_D.zero_grad()
            self.forward_D()
            self.backward_D()
            self.optimizer_D.step()

    def get_current_losses(self):
        l = {'G_loss': self.G_loss.item(), 'G_loss_ae': self.G_loss_ae.item()}
        if self.opt.pretrain_network is False:
            l.update({'G_loss_adv':        self.G_loss_adv.item(),
                      'G_loss_vgg':        self.G_loss_vgg.item(),
                      'G_loss_vgg_align':  self.vggloss.align_loss.item(),
                      'G_loss_vgg_guided': self.vggloss.guided_loss.item(),
                      'G_loss_vgg_fm':     self.vggloss.fm_vgg_loss.item(),
                      'D_loss':            self.D_loss.item(),
                      'G_loss_fm_dis':     self.G_loss_fm_dis.item()})
        return l

    def get_current_visuals(self):
        return {'input':     self.im_in.cpu().detach().numpy(), 'gt': self.gt.cpu().detach().numpy(),
                'completed': self.completed.cpu().detach().numpy()}

    def get_current_visuals_tensor(self):
        return {'input':     self.im_in.cpu().detach(), 'gt': self.gt.cpu().detach(),
                'completed': self.completed.cpu().detach()}

    def evaluate(self, im_in, mask):
        im_in = torch.from_numpy(im_in).type(torch.FloatTensor).cuda() / 127.5 - 1
        mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda()
        im_in = im_in * (1 - mask)
        xin = torch.cat((im_in, mask), 1)
        ret = self.netDFBN(xin) * mask + im_in * (1 - mask)
        ret = (ret.cpu().detach().numpy() + 1) * 127.5
        return ret.astype(np.uint8)

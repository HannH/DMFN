import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from model.layer import VGG19FeatLayer
from functools import reduce


class WGANLoss(nn.Module):
    def __init__(self):
        super(WGANLoss, self).__init__()

    def __call__(self, input, target):
        d_loss = (input - target).mean()
        g_loss = -input.mean()
        return {'g_loss': g_loss, 'd_loss': d_loss}


def gradient_penalty(xin, yout, mask=None):
    gradients = autograd.grad(yout, xin, create_graph=True,
                              grad_outputs=torch.ones(yout.size()).cuda(), retain_graph=True, only_inputs=True)[0]
    if mask is not None:
        gradients = gradients * mask
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def random_interpolate(gt, pred):
    batch_size = gt.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).cuda()
    # alpha = alpha.expand(gt.size()).cuda()
    interpolated = gt * alpha + pred * (1 - alpha)
    return interpolated


class VGGLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(VGGLoss, self).__init__()
        self.featlayer = featlayer()
        for k, v in self.featlayer.named_parameters():
            v.requires_grad = False
        self.self_guided_layers = ['relu1_1', 'relu2_1']
        self.feat_vgg_layers = ['relu{}_1'.format(x + 1) for x in range(5)]
        self.lambda_loss = 25
        self.gamma_loss = 1
        self.align_loss, self.guided_loss, self.fm_vgg_loss = None, None, None
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.coord_y, self.coord_x = torch.meshgrid(torch.arange(-1, 1, 1 / 8), torch.arange(-1, 1, 1 / 8))
        self.coord_y, self.coord_x = self.coord_y.cuda(), self.coord_x.cuda()

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def calc_align_loss(self, gen, tar):
        def sum_u_v(x):
            area = x.shape[-2] * x.shape[-1]
            return torch.sum(x.view(-1, area), -1) + 1e-7

        sum_gen = sum_u_v(gen)
        sum_tar = sum_u_v(tar)
        c_u_k = sum_u_v(self.coord_x * tar) / sum_tar
        c_v_k = sum_u_v(self.coord_y * tar) / sum_tar
        c_u_k_p = sum_u_v(self.coord_x * gen) / sum_gen
        c_v_k_p = sum_u_v(self.coord_y * gen) / sum_gen
        out = F.mse_loss(torch.stack([c_u_k, c_v_k], -1), torch.stack([c_u_k_p, c_v_k_p], -1), reduction='mean')
        return out

    def forward(self, gen, tar, mask_guidance, weight_fn):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)

        guided_loss_list = []
        mask_guidance = mask_guidance.unsqueeze(1)
        for layer in self.self_guided_layers:
            guided_loss_list += [F.l1_loss(gen_vgg_feats[layer] * mask_guidance, tar_vgg_feats[layer] * mask_guidance, reduction='sum') * weight_fn(tar_vgg_feats[layer])]
            mask_guidance = self.avg_pool(mask_guidance)
        self.guided_loss = reduce(lambda x, y: x + y, guided_loss_list)

        content_loss_list = [F.l1_loss(gen_vgg_feats[layer], tar_vgg_feats[layer], reduction='sum') * weight_fn(tar_vgg_feats[layer]) for layer in self.feat_vgg_layers]
        self.fm_vgg_loss = reduce(lambda x, y: x + y, content_loss_list)

        self.align_loss = self.calc_align_loss(gen_vgg_feats['relu4_1'], tar_vgg_feats['relu4_1'])

        return self.gamma_loss * self.align_loss + self.lambda_loss * (self.guided_loss + self.fm_vgg_loss)


class StyleLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer, style_layers=None):
        super(StyleLoss, self).__init__()
        self.featlayer = featlayer()
        if style_layers is not None:
            self.feat_style_layers = style_layers
        else:
            self.feat_style_layers = {'relu2_2': 1.0, 'relu3_2': 1.0, 'relu4_2': 1.0}

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        feats = x.view(b * c, h * w)
        g = torch.mm(feats, feats.t())
        return g.div(b * c * h * w)

    def _l1loss(self, gen, tar):
        return torch.abs(gen - tar).mean()

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self._l1loss(self.gram_matrix(gen_vgg_feats[layer]), self.gram_matrix(tar_vgg_feats[layer])) for
                           layer in self.feat_style_layers]
        style_loss = reduce(lambda x, y: x + y, style_loss_list)
        return style_loss


class ContentLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer, content_layers=None):
        super(ContentLoss, self).__init__()
        self.featlayer = featlayer()
        if content_layers is not None:
            self.feat_content_layers = content_layers
        else:
            self.feat_content_layers = {'relu4_2': 1.0}

    def _l1loss(self, gen, tar):
        return torch.abs(gen - tar).mean()

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        content_loss_list = [self.feat_content_layers[layer] * self._l1loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for
                             layer in self.feat_content_layers]
        content_loss = reduce(lambda x, y: x + y, content_loss_list)
        return content_loss


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        h_x, w_x = x.size()[2:]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :h_x - 1, :])
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :w_x - 1])
        loss = torch.sum(h_tv) + torch.sum(w_tv)
        return loss

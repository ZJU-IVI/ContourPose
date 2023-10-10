import torch
from torch import nn
from resnet import resnet18
import torch.nn.functional as F


class ContourPose(torch.nn.Module):
    def __init__(self,
                 fcdim=256, s16dim=256, s8dim=128, s4dim=64, s2dim=64, raw_dim=64,
                 seg_dim=2, feature_dim=64, heatmap_dim=8, edge_dim=1,
                 cat=True, dropout=0.1, sigma=100):
        super(ContourPose, self).__init__()

        self.sigma = sigma

        self.alpha = 0.99
        self.gamma = 2
        self.cat = cat  # Ture
        self.dropout = dropout  # 0.1
        self.img_convs = torch.nn.ModuleList()
        self.seg_dim = seg_dim  # 2
        self.feature_dim = feature_dim  # 64
        self.heatmap_dim = heatmap_dim
        self.edge_dim = edge_dim
        self.loss_fn = nn.MSELoss()  #
        self.seg_loss = nn.BCEWithLogitsLoss()

        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=16,
                               remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # The second encoder
        # x16s -> 256
        self.conv16s = nn.Sequential(
            nn.Conv2d(256 + fcdim, s16dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s16dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up16sto8s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + s16dim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv_raw = nn.Sequential(
            # input channel
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
        )
        self.conv_heatmap = nn.Sequential(
            nn.Conv2d(raw_dim, heatmap_dim, 1, 1)

        )
        self.conv_edge = nn.Sequential(
            nn.Conv2d(raw_dim, edge_dim, 1, 1)
        )

        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

    def weighted_cross_entropy_loss(self, pred_contour, target):
        """ Calculate sum of weighted cross entropy loss. """
        mask = (target > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3], keepdim=True).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        weight = torch.zeros_like(mask)
        weight.masked_scatter_(target > 0.5,
                               torch.ones_like(target) * num_neg / (num_pos + num_neg))
        weight.masked_scatter_(target <= 0.5,
                               torch.ones_like(target) * num_pos / (num_pos + num_neg))
        losses = F.binary_cross_entropy_with_logits(
            pred_contour.float(), target.float(), weight=weight, reduction='none')
        loss = torch.sum(losses) / b
        return loss

    def forward(self, x, heatmap = None, target_contour = None):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)
        fm1 = self.conv16s(torch.cat([xfc, x16s], 1))
        fm1 = self.up16sto8s(fm1)
        fm1 = self.conv8s(torch.cat([fm1, x8s], 1))
        fm1 = self.up8sto4s(fm1)
        if fm1.shape[2] == 136:
            fm1 = nn.functional.interpolate(fm1, (135, 180), mode='bilinear', align_corners=False)

        fm1 = self.conv4s(torch.cat([fm1, x4s], 1))
        fm1 = self.up4sto2s(fm1)

        fm1 = self.conv2s(torch.cat([fm1, x2s], 1))
        fm1 = self.up2storaw(fm1)
        fm1 = self.conv_raw(torch.cat([fm1, x], 1))
        fm1 = self.conv_heatmap(fm1)

        pred_heatmap = fm1
        fm2 = self.conv16s(torch.cat([xfc, x16s], 1))
        fm2 = self.up16sto8s(fm2)
        fm2 = self.conv8s(torch.cat([fm2, x8s], 1))
        fm2 = self.up8sto4s(fm2)
        if fm2.shape[2] == 136:
            fm2 = nn.functional.interpolate(fm2, (135, 180), mode='bilinear', align_corners=False)

        fm2 = self.conv4s(torch.cat([fm2, x4s], 1))
        fm2 = self.up4sto2s(fm2)

        fm2 = self.conv2s(torch.cat([fm2, x2s], 1))
        fm2 = self.up2storaw(fm2)
        fm2 = self.conv_raw(torch.cat([fm2, x], 1))
        fm2 = self.conv_edge(fm2)
        pred_contour = fm2

        if self.training:
            loss_fn = nn.MSELoss()
            heatmap_loss = loss_fn(pred_heatmap, heatmap) * 1000
            # contour_loss = self.weighted_cross_entropy_loss(pred_contour, target_contour)
            contour_loss = self.seg_loss(pred_contour.float(), target_contour.float())

            loss = {}
            loss["heatmap_loss"] = heatmap_loss
            loss["contour_loss"] = contour_loss
            return loss
        else:
            return pred_heatmap, pred_contour

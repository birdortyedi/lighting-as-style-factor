"""
Official implementation of DeNIM (Deterministic Neural Illumination Mapping with Style WB Backbone)

"Deterministic Neural Illumination Mapping for Efficient Auto-White Balance Correction"
Furkan Kınlı, Doğa Yılmaz, Barış Özcan and Furkan Kıraç
Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, 1139-1147
https://openaccess.thecvf.com/content/ICCV2023W/RCV/html/Kinli_Deterministic_Neural_Illumination_Mapping_for_Efficient_Auto-White_Balance_Correction_ICCVW_2023_paper.html
"""


import torch
from torch import nn
from torchvision.models import vgg16

from kornia.geometry.transform import resize

from style_wb import IFRNetv2


class DeNIM(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.empty((3, k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((k, 3)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.Q)
        self.k = k
        
    def forward(self, I, T):
        bs, _, H, W = I.shape
        x = torch.flatten(I, start_dim=2).transpose(1, 2)
        out = x @ self.P @ T.view(bs, self.k, self.k) @ self.Q
        out = out.view(bs, H, W, -1).permute(0, 3, 1, 2)

        return out
    
    
class DeNIM_to_Canon(nn.Module):
    def __init__(self, k, ch: int = 3) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.empty((ch, k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((k, ch)), requires_grad=True)
        self.R = nn.Parameter(torch.empty((ch, 3)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.Q)
        torch.nn.init.kaiming_normal_(self.R)
        self.k = k
        
    def forward(self, I, T):
        bs, _, H, W = I.shape
        x = torch.flatten(I, start_dim=2).transpose(1, 2)
        out = x @ self.P @ T.view(bs, self.k, self.k) @ self.Q @ self.R
        out = out.view(bs, H, W, -1).permute(0, 3, 1, 2)

        return out
    

class DeNIM_wo_Fusion(nn.Module):
    def __init__(self, k, ch: int = 3) -> None:
        super().__init__()
        self.P = nn.Parameter(torch.empty((ch, k)), requires_grad=True)
        self.T = nn.Parameter(torch.empty((k, k)), requires_grad=True)
        self.Q = nn.Parameter(torch.empty((k, ch)), requires_grad=True)
        torch.nn.init.kaiming_normal_(self.P)
        torch.nn.init.kaiming_normal_(self.T)
        torch.nn.init.kaiming_normal_(self.Q)
        self.k = k
        
    def forward(self, I):
        bs, _, H, W = I.shape
        x = torch.flatten(I, start_dim=2).transpose(1, 2)
        out = x @ self.P @ self.T @ self.Q
        out = out.view(bs, H, W, -1).permute(0, 3, 1, 2)

        return out


class AWBEncoder(nn.Module):
    def __init__(
        self,
        sz,
        k,
        backbone_inchnls: int,
        backbone_ps: int,
        backbone_weights: str = None,
        device: str = "cuda"
    ) -> None:
        super().__init__()
        self.sz = sz
        self.backbone_inchnls = backbone_inchnls
        self.backbone_ps = backbone_ps
        self.backbone_weights = backbone_weights
        self.device = device
        self._1x1conv = torch.nn.Conv2d(256, 1, 1)
        self.act = torch.nn.GELU()
        
        self._init_backbone()
        self.D = nn.Linear(in_features=1024, out_features=k*k)

    def _init_backbone(self):
        vgg_feats = vgg16(pretrained=True).features.eval().cuda()
        vgg_feats = nn.Sequential(*[module for module in vgg_feats][:35])
        self.backbone = IFRNetv2(vgg_feats, in_channels=self.backbone_inchnls, base_n_channels=32, input_size=self.backbone_ps).cuda()  # IFRNetv2
        
        if self.backbone_weights is None:
            return
        state_dict = torch.load(self.backbone_weights, map_location=self.device)
        self.backbone.load_state_dict(state_dict)
    
    def forward(self, x):
        # x, shape: B x C x H x W
        with torch.no_grad():
            out = resize(x, (self.sz, self.sz), interpolation='bilinear')  # shape: B x C x sz x sz
            out = self.backbone.forward_encoder(out)  # shape: B x 256 x 32 x 32 
        out = self.act(self._1x1conv(out))  # shape: B x 1 x 32 x 32
        out = torch.flatten(out, start_dim=1)  # shape: B x 1024

        return self.D(out)  # shape: B x k*k


if __name__ == "__main__":
    k, ch, sz, ps = 32, 9, 256, 64
    dncm_to_canon = DeNIM_to_Canon(k, ch).cuda()
    dncm_wo_fusion = DeNIM_wo_Fusion(k).cuda()
    awb_encoder = AWBEncoder(sz, k, ch, ps, None).cuda()

    I = torch.randn(1, ch, 2048, 2048).cuda()
    with torch.no_grad():
        d = awb_encoder(I)
        canon = dncm_to_canon(I, d)
        out = dncm_wo_fusion(canon)
    print(out.shape)

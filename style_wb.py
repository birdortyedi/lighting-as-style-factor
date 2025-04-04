"""
Official implementation of Style WB Network (IFRNet variant for white balance correction)

"Modeling the Lighting in Scenes as Style for Auto White-Balance Correction"
Furkan Kınlı, Doğa Yılmaz, Barış Özcan and Furkan Kıraç
Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2023, 4903-4913
https://openaccess.thecvf.com/content/WACV2023/html/Kinli_Modeling_the_Lighting_in_Scenes_As_Style_for_Auto_White-Balance_WACV_2023_paper.html
"""


import torch
from torch import nn

from kornia.geometry.transform import resize


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def forward(self, x, y):
        pass

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def init_weights(self, init_type='xavier', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Destyler(nn.Module):
    def __init__(self, in_features, num_features, num_layers=2):
        super(Destyler, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_features, num_features),
            *[nn.Linear(num_features, num_features) for _ in range(num_layers-1)]
        )

    def forward(self, x):
        return self.fcs(x)


class ResBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False):
        super(ResBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.n2 = nn.BatchNorm2d(channels_out)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.lrelu1(out)
        out = self.conv2(out)
        # out = self.n2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.lrelu2(out)

        return out


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        ch = y.size(1)
        sigma, mu = torch.split(y.unsqueeze(-1).unsqueeze(-1), [ch // 2, ch // 2], dim=1)

        x_mu = x.mean(dim=[2, 3], keepdim=True)
        x_sigma = x.std(dim=[2, 3], keepdim=True)

        return sigma * ((x - x_mu) / x_sigma) + mu


class DestyleResBlock(nn.Module):
    def __init__(self, channels_out, kernel_size, channels_in=None, stride=1, dilation=1, padding=1, use_dropout=False, style_proj_n_ch=128):
        super(DestyleResBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == channels_out:
            channels_in = channels_out
            self.projection = None
        else:
            self.projection = nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=stride, dilation=1)
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.adain = AdaIN()
        self.style_projector = nn.Linear(style_proj_n_ch, channels_out * 2)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, feat):
        residual = x
        out = self.conv1(x)
        out = self.lrelu1(out)
        out = self.conv2(out)
        _, _, h, w = out.size()
        style_proj = self.style_projector(feat)
        out = self.adain(out, style_proj)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out = out + residual
        out = self.lrelu2(out)

        return out


class IFRNetv2(BaseNetwork):
    def __init__(self, vgg_feats, input_size, base_n_channels, in_channels=3, out_channels=3):
        super(IFRNetv2, self).__init__()
        self.vgg16_feat = vgg_feats
        self.input_size = input_size
        self.destyler = Destyler(in_features=input_size * input_size // 2, num_features=base_n_channels // 2)  # from vgg features

        style_proj_num_ch = (base_n_channels // 2) * (in_channels // 3)
        self.ds_res1 = DestyleResBlock(channels_in=in_channels, channels_out=base_n_channels, style_proj_n_ch=style_proj_num_ch, kernel_size=5, stride=1, padding=2)
        self.ds_res2 = DestyleResBlock(channels_in=base_n_channels, channels_out=base_n_channels * 2, style_proj_n_ch=style_proj_num_ch, kernel_size=3, stride=2, padding=1)
        self.ds_res3 = DestyleResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, style_proj_n_ch=style_proj_num_ch, kernel_size=3, stride=1, padding=1)
        self.ds_res4 = DestyleResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 4, style_proj_n_ch=style_proj_num_ch, kernel_size=3, stride=2, padding=1)
        self.ds_res5 = DestyleResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 8, style_proj_n_ch=style_proj_num_ch, kernel_size=3, stride=2, padding=1)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

        self.res2 = ResBlock(channels_in=base_n_channels * 8, channels_out=base_n_channels * 4, kernel_size=3, stride=1, padding=1)
        self.res3 = ResBlock(channels_in=base_n_channels * 4, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.res4 = ResBlock(channels_in=base_n_channels * 2, channels_out=base_n_channels * 2, kernel_size=3, stride=1, padding=1)
        self.res5 = ResBlock(channels_in=base_n_channels * 2, channels_out=out_channels, kernel_size=3, stride=1, padding=1)

        self.init_weights(init_type="normal", gain=0.02)

    def extract_vgg_feat(self, x):
        with torch.no_grad():
            vgg_feat = self.vgg16_feat(resize(x, (self.input_size, self.input_size), align_corners=True))
        b_size, ch, h, w = vgg_feat.size()
        vgg_feat = vgg_feat.view(b_size, ch * h * w)
        return self.destyler(vgg_feat)

    def forward(self, x):
        vgg_feat = self.extract_vgg_feat(x[:, :3, :, :])
        for i in range(1, int(x.shape[1] // 3)):
            vgg_feat_patch = self.extract_vgg_feat(x[:, (i * 3):3 + (i * 3), :, :])
            vgg_feat = torch.cat([vgg_feat, vgg_feat_patch], dim=1)

        out = self.ds_res1(x, vgg_feat)
        out = self.ds_res2(out, vgg_feat)
        out = self.ds_res3(out, vgg_feat)
        out = self.ds_res4(out, vgg_feat)
        out = self.ds_res5(out, vgg_feat)

        out = self.upsample(out)
        out = self.res2(out)
        out = self.upsample(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.upsample(out)
        out = self.res5(out)

        return out

    def forward_encoder(self, x):
        vgg_feat = self.extract_vgg_feat(x[:, :3, :, :])
        for i in range(1, int(x.shape[1] // 3)):
            vgg_feat_patch = self.extract_vgg_feat(x[:, (i * 3):3 + (i * 3), :, :])
            vgg_feat = torch.cat([vgg_feat, vgg_feat_patch], dim=1)

        out = self.ds_res1(x, vgg_feat)
        out = self.ds_res2(out, vgg_feat)
        out = self.ds_res3(out, vgg_feat)
        out = self.ds_res4(out, vgg_feat)
        out = self.ds_res5(out, vgg_feat)

        return out

        
if __name__ == '__main__':
    from torchvision.models import vgg16
    vgg_feats = vgg16(pretrained=True).features.eval().cuda()
    vgg_feats = nn.Sequential(*[module for module in vgg_feats][:35])
    # 5 WB settings with 3 channels for each setting
    net = IFRNetv2(vgg_feats, in_channels=15, base_n_channels=32, input_size=64, out_channels=5).cuda()
    x = torch.rand(8, 15, 256, 256).cuda()  

    # forward pass of WB correction network
    weights = net(x)
    weights = torch.clamp(weights, -1000, 1000)
    weights = nn.Softmax(dim=1)(weights)
    
    # blend the weights with their corresponding WB settings
    out_img = torch.unsqueeze(weights[:, 0, :, :], dim=1) * x[:, :3, :, :]
    for i in range(1, int(x.shape[1] // 3)):
        out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * x[:, (i * 3):3 + (i * 3), :, :]
    print(out_img.shape, weights.shape)

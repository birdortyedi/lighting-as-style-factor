"""
Official implementation of FDM loss

"Feature distribution statistics as a loss objective for robust white balance correction"
Furkan Kınlı and Furkan Kıraç
Machine Vision and Applications, 36 (3), 1-20
https://link.springer.com/article/10.1007/s00138-025-01680-1
"""


import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoImageProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def attn_cosine_sim(x, eps=1e-08):
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor

    return sim_matrix


class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(-1, patch_num, 3, head_num, embedding_dim // head_num).permute(0, 2, 3, 1, 4)[: ,1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(-1, patch_num, 3, head_num, embedding_dim // head_num).permute(0, 2, 3, 1, 4)[: ,2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        b, h, t, d = keys.shape
        concatenated_keys = keys.transpose(1, 2).reshape(b, t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys)
        return ssim_map


def exact_feature_distribution_matching(content_feat, style_feat):
    _, index_content = torch.sort(content_feat)
    value_style, _ = torch.sort(style_feat)
    inverse_index = index_content.argsort(-1)
    loss = F.mse_loss(content_feat, value_style.gather(-1, inverse_index))
    return loss

efdm_loss = exact_feature_distribution_matching


def apply_wb(org_img, pred):
    pred_rgb = torch.zeros_like(org_img) 
    pred_rgb[:,1,:,:] = org_img[:,1,:,:]
    pred_rgb[:,0,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,0,:,:])   # R = G * (R/G)
    pred_rgb[:,2,:,:] = org_img[:,1,:,:] * torch.exp(pred[:,1,:,:])   # B = G * (B/G)

    return pred_rgb


class FDMLoss(nn.Module):
    def __init__(self, white_level):
        super().__init__()
        self.white_level = white_level
        self.processor = AutoImageProcessor.from_pretrained('facebook/dino-vits8')
        self.extractor = VitExtractor(model_name="dino_vits8", device=device)
      
        self.cls_warmup = 10
        self.lambdas = dict(
            lambda_cls=10,
            lambda_ssim=0,
            lambda_identity=0
        )

    def update_lambda_config(self, step):
        if step == self.cls_warmup:
            self.lambdas['lambda_ssim'] = 1.0
            self.lambdas['lambda_identity'] = 1.0

    def forward(self, y, inputs):
        self.update_lambda_config(inputs['step'])
        losses = {
            "loss_ssim": 0,
            "loss_cls": 0,
            "loss_identity": 0,
            "loss": 0
        }
        loss_G = 0.0
        pred_rgb = torch.clamp(apply_wb(inputs['input_rgb'], y) / self.white_level, 0, 1)
        pred_rgb = self.processor(pred_rgb, do_rescale=False, use_fast=True, return_tensors="pt")["pixel_values"].to(device)
        gt_rgb = torch.clamp(apply_wb(inputs['input_rgb'], inputs['gt_uv']) / self.white_level, 0, 1)
        gt_rgb = self.processor(gt_rgb, do_rescale=False, use_fast=True, return_tensors="pt")["pixel_values"].to(device)
        x_rgb = self.processor(torch.clamp(inputs['input_rgb'] / self.white_level, 0, 1), do_rescale=False, use_fast=True, return_tensors="pt")["pixel_values"].to(device)

        if self.lambdas['lambda_ssim'] > 0:
            losses['loss_ssim'] = self.calculate_global_ssim_loss(pred_rgb, x_rgb)
            loss_G += losses['loss_ssim'] * self.lambdas['lambda_ssim']

        if self.lambdas['lambda_identity'] > 0:
            losses['loss_identity'] = self.calculate_global_id_loss(pred_rgb, x_rgb)
            loss_G += losses['loss_identity'] * self.lambdas['lambda_identity']

        if self.lambdas['lambda_cls'] > 0:
            losses['loss_cls'] = self.calculate_crop_cls_loss(pred_rgb, gt_rgb)
            loss_G += losses['loss_cls'] * self.lambdas['lambda_cls']

        losses['loss'] = loss_G
        return losses

    def calculate_global_ssim_loss(self, outputs, inputs):
        with torch.no_grad():
            target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(inputs, layer_num=11)
        keys_ssim = self.extractor.get_keys_self_sim_from_input(outputs, layer_num=11)
        loss = F.mse_loss(keys_ssim, target_keys_self_sim)

        return loss

    def calculate_crop_cls_loss(self, outputs, inputs):
        with torch.no_grad():
            target_cls_token = self.extractor.get_feature_from_input(inputs)[-1][:, 0, :]
        cls_token = self.extractor.get_feature_from_input(outputs)[-1][:, 0, :]
        loss = efdm_loss(cls_token, target_cls_token)

        return loss

    def calculate_global_id_loss(self, outputs, inputs):
        with torch.no_grad():
            keys_a = self.extractor.get_keys_from_input(inputs, 11)
        keys_b = self.extractor.get_keys_from_input(outputs, 11)
        loss = F.mse_loss(keys_a, keys_b)

        return loss

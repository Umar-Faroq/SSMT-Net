from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from collections import OrderedDict
from torch.nn import Conv2d, Dropout, LayerNorm, Linear, Softmax
from torch.nn.modules.utils import _pair
from torch.utils.data import Dataset


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def np2th(weights: np.ndarray, conv: bool = False) -> torch.Tensor:
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis: bool):
        super().__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)
        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*context_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(self, config, vis: bool):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights: Dict[str, np.ndarray], n_block: int) -> None:
        root = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[f"{root}/{ATTENTION_Q}/kernel"]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[f"{root}/{ATTENTION_K}/kernel"]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[f"{root}/{ATTENTION_V}/kernel"]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[f"{root}/{ATTENTION_OUT}/kernel"]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[f"{root}/{ATTENTION_Q}/bias"]).view(-1)
            key_bias = np2th(weights[f"{root}/{ATTENTION_K}/bias"]).view(-1)
            value_bias = np2th(weights[f"{root}/{ATTENTION_V}/bias"]).view(-1)
            out_bias = np2th(weights[f"{root}/{ATTENTION_OUT}/bias"]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[f"{root}/{FC_0}/kernel"]).t()
            mlp_weight_1 = np2th(weights[f"{root}/{FC_1}/kernel"]).t()
            mlp_bias_0 = np2th(weights[f"{root}/{FC_0}/bias"]).t()
            mlp_bias_1 = np2th(weights[f"{root}/{FC_1}/bias"]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[f"{root}/{ATTENTION_NORM}/scale"]))
            self.attention_norm.bias.copy_(np2th(weights[f"{root}/{ATTENTION_NORM}/bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[f"{root}/{MLP_NORM}/scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[f"{root}/{MLP_NORM}/bias"]))


class Encoder(nn.Module):
    def __init__(self, config, vis: bool):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList([Block(config, vis) for _ in range(config.transformer["num_layers"])])
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class StdConv2d(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        return F.conv2d(x, (w - m) / torch.sqrt(v + 1e-5), self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(cin: int, cout: int, stride: int = 1, groups: int = 1) -> StdConv2d:
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)


def conv1x1(cin: int, cout: int, stride: int = 1) -> StdConv2d:
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False)


class PreActBottleneck(nn.Module):
    def __init__(self, cin: int, cout: Optional[int] = None, cmid: Optional[int] = None, stride: int = 1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)

    def load_from(self, weights: Dict[str, np.ndarray], n_block: str, n_unit: str) -> None:
        conv1_weight = np2th(weights[f"{n_block}/{n_unit}/conv1/kernel"], conv=True)
        conv2_weight = np2th(weights[f"{n_block}/{n_unit}/conv2/kernel"], conv=True)
        conv3_weight = np2th(weights[f"{n_block}/{n_unit}/conv3/kernel"], conv=True)

        gn1_weight = np2th(weights[f"{n_block}/{n_unit}/gn1/scale"])
        gn1_bias = np2th(weights[f"{n_block}/{n_unit}/gn1/bias"])
        gn2_weight = np2th(weights[f"{n_block}/{n_unit}/gn2/scale"])
        gn2_bias = np2th(weights[f"{n_block}/{n_unit}/gn2/bias"])
        gn3_weight = np2th(weights[f"{n_block}/{n_unit}/gn3/scale"])
        gn3_bias = np2th(weights[f"{n_block}/{n_unit}/gn3/bias"])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)
        self.gn1.weight.copy_(gn1_weight)
        self.gn1.bias.copy_(gn1_bias)
        self.gn2.weight.copy_(gn2_weight)
        self.gn2.bias.copy_(gn2_bias)
        self.gn3.weight.copy_(gn3_weight)
        self.gn3.bias.copy_(gn3_bias)

        if hasattr(self, "downsample"):
            proj_conv_weight = np2th(weights[f"{n_block}/{n_unit}/conv_proj/kernel"], conv=True)
            proj_gn_weight = np2th(weights[f"{n_block}/{n_unit}/gn_proj/scale"])
            proj_gn_bias = np2th(weights[f"{n_block}/{n_unit}/gn_proj/bias"])
            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight)
            self.gn_proj.bias.copy_(proj_gn_bias)


class ResNetV2(nn.Module):
    def __init__(self, block_units: Sequence[int], width_factor: int):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ("gn", nn.GroupNorm(32, width, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ("block1", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f"unit{i:d}", PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in range(2, block_units[0] + 1)]
            ))),
            ("block2", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f"unit{i:d}", PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in range(2, block_units[1] + 1)]
            ))),
            ("block3", nn.Sequential(OrderedDict(
                [("unit1", PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f"unit{i:d}", PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in range(2, block_units[2] + 1)]
            ))),
        ]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert 0 < pad < 3
                feat = torch.zeros((b, x.size(1), right_size, right_size), device=x.device)
                feat[:, :, 0:x.size(2), 0:x.size(3)] = x
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class Embeddings(nn.Module):
    def __init__(self, config, img_size: int, in_channels: int = 3):
        super().__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet["num_layers"], width_factor=config.resnet["width_factor"])
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        features = None
        if self.hybrid:
            x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        embeddings = self.dropout(x + self.position_embeddings)
        return embeddings, features


class Transformer(nn.Module):
    def __init__(self, config, img_size: int, vis: bool):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[torch.Tensor]]]:
        embedding_output, features = self.embeddings(inputs)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, padding: int = 0, stride: int = 1, use_batchnorm: bool = True):
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        bn = nn.BatchNorm2d(out_ch) if use_batchnorm else nn.Identity()
        super().__init__(conv, bn, nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int = 0, use_batchnorm: bool = True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(out_ch, out_ch, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SegmentationHead(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, upsampling: int = 1):
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv, up)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels, kernel_size=3, padding=1, use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels = config.skip_channels
        if config.n_skip != 0:
            for i in range(4 - config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, skip_ch=skip_ch)
            for in_ch, out_ch, skip_ch in zip(in_channels, out_channels, skip_channels)
        ])

    def forward(self, hidden_states: torch.Tensor, features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        b, n_patch, hidden = hidden_states.size()
        h = w = int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(b, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = decoder_block(x, skip=skip)
        return x


class ReconstructionDecoder(nn.Module):
    def __init__(self, config, out_channels: int = 3):
        super().__init__()
        self.decoder = DecoderCup(config)
        self.head = nn.Sequential(
            Conv2dReLU(config.decoder_channels[-1], config.decoder_channels[-1], kernel_size=3, padding=1),
            nn.Conv2d(config.decoder_channels[-1], out_channels, kernel_size=1),
        )

    def forward(self, encoded: torch.Tensor, features: Optional[List[torch.Tensor]]) -> torch.Tensor:
        return self.head(self.decoder(encoded, features))


class SpatialTransformerDecoder(nn.Module):
    def __init__(self, config, out_channels: int, query_size: int = 28):
        super().__init__()
        hidden_size = config.hidden_size
        self.query_size = query_size
        self.query_proj = nn.Conv2d(config.decoder_channels[-1], hidden_size, kernel_size=1)
        self.memory_proj = nn.Linear(hidden_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=config.transformer["num_heads"],
            dim_feedforward=config.transformer["mlp_dim"],
            dropout=config.transformer["dropout_rate"],
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.out = nn.Sequential(
            Conv2dReLU(hidden_size, config.decoder_channels[-1], kernel_size=3, padding=1),
            nn.Conv2d(config.decoder_channels[-1], out_channels, kernel_size=1),
        )

    def forward(self, local_feature: torch.Tensor, encoded_tokens: torch.Tensor) -> torch.Tensor:
        b, c, h, w = local_feature.shape
        query_feature = F.adaptive_avg_pool2d(local_feature, output_size=(self.query_size, self.query_size))
        queries = self.query_proj(query_feature).flatten(2).transpose(1, 2)
        memory = self.memory_proj(encoded_tokens)
        decoded = self.decoder(tgt=queries, memory=memory)
        decoded = decoded.transpose(1, 2).contiguous().view(b, -1, self.query_size, self.query_size)
        decoded = self.out(decoded)
        return F.interpolate(decoded, size=(h, w), mode="bilinear", align_corners=False)


class SizeEstimator(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, encoded_tokens: torch.Tensor) -> torch.Tensor:
        pooled = encoded_tokens.mean(dim=1)
        return self.head(pooled)


class SSMTNet(nn.Module):
    def __init__(self, config, img_size: int = 224, in_channels: int = 3, vis: bool = False):
        super().__init__()
        self.transformer = Transformer(config, img_size, vis)
        self.cnn_decoder = DecoderCup(config)
        self.gland_cnn_head = SegmentationHead(config.decoder_channels[-1], 1, kernel_size=3)
        self.nodule_cnn_head = SegmentationHead(config.decoder_channels[-1], 1, kernel_size=3)
        self.gland_transformer_decoder = SpatialTransformerDecoder(config, out_channels=1, query_size=28)
        self.nodule_transformer_decoder = SpatialTransformerDecoder(config, out_channels=1, query_size=28)
        self.reconstructor = ReconstructionDecoder(config, out_channels=in_channels)
        self.size_estimator = SizeEstimator(config.hidden_size)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded, _, features = self.transformer(x)
        local_feature = self.cnn_decoder(encoded, features)

        gland_local = self.gland_cnn_head(local_feature)
        nodule_local = self.nodule_cnn_head(local_feature)
        gland_global = self.gland_transformer_decoder(local_feature, encoded)
        nodule_global = self.nodule_transformer_decoder(local_feature, encoded)

        return {
            "gland_local": gland_local,
            "nodule_local": nodule_local,
            "gland_logits": gland_local + gland_global,
            "nodule_logits": nodule_local + nodule_global,
            "reconstruction": self.reconstructor(encoded, features),
            "size": self.size_estimator(encoded),
            "decoder_feature": local_feature,
            "tokens": encoded,
        }

    def load_from(self, weights: Dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(
                np2th(weights["embedding/kernel"], conv=True)
            )
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                ntok_new = posemb_new.size(1)
                posemb_grid = posemb[0, 1:] if posemb.size(1) != ntok_new else posemb[0]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb_grid))
            for block_index, block in enumerate(self.transformer.encoder.layer):
                block.load_from(weights, n_block=block_index)


@dataclass
class SSMTConfig:
    hidden_size: int = 768
    transformer: Dict[str, float] = None
    patches: Dict[str, Tuple[int, int]] = None
    resnet: Dict[str, Sequence[int]] = None
    classifier: str = "seg"
    representation_size: Optional[int] = None
    pretrained_path: Optional[str] = None
    patch_size: int = 16
    decoder_channels: Tuple[int, int, int, int] = (256, 128, 64, 16)
    skip_channels: List[int] = None
    n_classes: int = 1
    n_skip: int = 3
    activation: str = "sigmoid"

    def __post_init__(self) -> None:
        if self.transformer is None:
            self.transformer = {
                "mlp_dim": 3072,
                "num_heads": 12,
                "num_layers": 12,
                "attention_dropout_rate": 0.0,
                "dropout_rate": 0.1,
            }
        if self.patches is None:
            self.patches = {"size": (16, 16), "grid": (14, 14)}
        if self.resnet is None:
            self.resnet = {"num_layers": (3, 4, 9), "width_factor": 1}
        if self.skip_channels is None:
            self.skip_channels = [512, 256, 64, 16]


def get_ssmt_r50_b16_config(img_size: int = 224, pretrained_path: Optional[str] = None) -> SSMTConfig:
    return SSMTConfig(
        patches={"size": (16, 16), "grid": (int(img_size / 16), int(img_size / 16))},
        pretrained_path=pretrained_path,
    )


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + dice_loss_from_logits(logits, targets)


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.mean(torch.sqrt((prediction - target) ** 2 + eps ** 2))


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)
    tn = ((1 - preds) * (1 - targets)).sum(dim=1)
    eps = 1e-7
    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()
    f1 = ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean().item()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    acc = ((tp + tn + eps) / (tp + tn + fp + fn + eps)).mean().item()
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "accuracy": acc}


def compute_size_target(mask: np.ndarray) -> float:
    mask_bin = (mask > 0.5).astype(np.float32)
    return float(mask_bin.mean())


def resize_mask(mask: np.ndarray, img_size: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_NEAREST)


def load_rgb(path: Path, img_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)


def load_mask(path: Path, img_size: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    mask = resize_mask(mask, img_size)
    return (mask > 127).astype(np.float32)


def load_tg3k_splits(json_path: Path) -> Dict[str, List[str]]:
    import json

    raw = json.loads(json_path.read_text())
    return {
        split: [f"{int(idx):04d}" for idx in values]
        for split, values in raw.items()
    }


class ThyroidMultiTaskDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        nodule_mask_paths: Sequence[Optional[Path]],
        gland_mask_paths: Sequence[Optional[Path]],
        img_size: int = 224,
        augment: bool = False,
        include_supervision: bool = True,
    ):
        self.image_paths = list(image_paths)
        self.nodule_mask_paths = list(nodule_mask_paths)
        self.gland_mask_paths = list(gland_mask_paths)
        self.img_size = img_size
        self.augment = augment
        self.include_supervision = include_supervision

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image = load_rgb(self.image_paths[index], self.img_size).astype(np.float32) / 255.0
        nodule_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        gland_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        if self.include_supervision:
            if self.nodule_mask_paths[index] is not None:
                nodule_mask = load_mask(self.nodule_mask_paths[index], self.img_size)
            if self.gland_mask_paths[index] is not None:
                gland_mask = load_mask(self.gland_mask_paths[index], self.img_size)

        if self.augment:
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                nodule_mask = np.flip(nodule_mask, axis=1).copy()
                gland_mask = np.flip(gland_mask, axis=1).copy()
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=0).copy()
                nodule_mask = np.flip(nodule_mask, axis=0).copy()
                gland_mask = np.flip(gland_mask, axis=0).copy()

        size_value = compute_size_target(nodule_mask)

        return {
            "image": torch.from_numpy(np.transpose(image, (2, 0, 1))).float(),
            "nodule_mask": torch.from_numpy(nodule_mask[None, ...]).float(),
            "gland_mask": torch.from_numpy(gland_mask[None, ...]).float(),
            "size_target": torch.tensor([size_value], dtype=torch.float32),
            "has_gland": torch.tensor([1.0 if self.gland_mask_paths[index] is not None else 0.0], dtype=torch.float32),
            "has_nodule": torch.tensor([1.0 if self.nodule_mask_paths[index] is not None else 0.0], dtype=torch.float32),
            "image_path": str(self.image_paths[index]),
        }


def build_tn3k_multitask_splits(
    dataset_root: Path,
    tg3k_root: Path,
    tg3k_split_json: Path,
) -> Dict[str, Dict[str, List[Optional[Path]]]]:
    tn3k_train_img = dataset_root / "trainval-image"
    tn3k_train_mask = dataset_root / "trainval-mask"
    tn3k_test_img = dataset_root / "test-image"
    tn3k_test_mask = dataset_root / "test-mask"
    tg3k_image_dir = tg3k_root / "thyroid-image"
    tg3k_mask_dir = tg3k_root / "thyroid-mask"

    tg3k_splits = load_tg3k_splits(tg3k_split_json)

    def path_index(folder: Path) -> Dict[str, Path]:
        return {path.stem: path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS}

    train_img_idx = path_index(tn3k_train_img)
    train_mask_idx = path_index(tn3k_train_mask)
    test_img_idx = path_index(tn3k_test_img)
    test_mask_idx = path_index(tn3k_test_mask)
    gland_img_idx = path_index(tg3k_image_dir)
    gland_mask_idx = path_index(tg3k_mask_dir)

    def collect(split_ids: Sequence[str], image_idx: Dict[str, Path], mask_idx: Dict[str, Path]) -> Dict[str, List[Optional[Path]]]:
        images, nodule_masks, gland_masks = [], [], []
        for sample_id in split_ids:
            if sample_id not in image_idx or sample_id not in mask_idx:
                continue
            images.append(image_idx[sample_id])
            nodule_masks.append(mask_idx[sample_id])
            gland_masks.append(gland_mask_idx.get(sample_id))
        return {"images": images, "nodule_masks": nodule_masks, "gland_masks": gland_masks}

    trainval_ids = set(train_img_idx) & set(train_mask_idx)
    test_ids = sorted(set(test_img_idx) & set(test_mask_idx))
    train_ids = [sample_id for sample_id in tg3k_splits["train"] if sample_id in trainval_ids]
    val_ids = [sample_id for sample_id in tg3k_splits["val"] if sample_id in trainval_ids]
    if not val_ids:
        cutoff = max(1, int(0.15 * len(trainval_ids)))
        ordered = sorted(trainval_ids)
        val_ids = ordered[:cutoff]
        train_ids = ordered[cutoff:]

    unlabeled_ids = [
        sample_id for sample_id in tg3k_splits["train"]
        if sample_id in gland_img_idx and sample_id not in set(test_ids) and sample_id not in set(val_ids)
    ]

    return {
        "train": collect(train_ids, train_img_idx, train_mask_idx),
        "val": collect(val_ids, train_img_idx, train_mask_idx),
        "test": collect(test_ids, test_img_idx, test_mask_idx),
        "unlabeled": {
            "images": [gland_img_idx[sample_id] for sample_id in unlabeled_ids],
            "nodule_masks": [None] * len(unlabeled_ids),
            "gland_masks": [None] * len(unlabeled_ids),
        },
    }

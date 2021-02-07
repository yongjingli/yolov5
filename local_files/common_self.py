# This file contains modules common to various models

import math

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, xyxy2xywh
from utils.plots import color_list

from models.common import Conv


class FocusNew(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(FocusNew, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, size, scale, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        sh = torch.tensor(x.shape)
        return F.interpolate(x, size=(int(sh[2]*self.scale), int(sh[3]*self.scale)), mode=self.mode, align_corners=self.align_corners)


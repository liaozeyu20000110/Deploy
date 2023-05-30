import torch
import torchvision.models as models
import torch.onnx
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import numpy as np

BATCH_SIZE = 64


if __name__ == '__main__':
    resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224)
    torch.onnx.export(resnext50_32x4d, dummy_input, "resnet50_onnx_model.onnx", verbose=False)

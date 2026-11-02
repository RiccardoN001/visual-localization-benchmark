from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from . import Retriever

# Official VGG16-NetVLAD Pitts30k weights converted from MATLAB, as distributed by hloc.
CHECKPOINT_URL = "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat"


def checkpoint_path():
    return Path(torch.hub.get_dir()) / "netvlad" / "VGG16-NetVLAD-Pitts30K.mat"


def download_checkpoint():
    path = checkpoint_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(CHECKPOINT_URL, str(path))
    return path


class NetVLADLayer(nn.Module):
    # Same operations and order as hloc/extractors/netvlad.py, which the MATLAB weights expect.
    def __init__(self, input_dim=512, K=64, score_bias=False, intranorm=True):
        super().__init__()
        self.score_proj = nn.Conv1d(input_dim, K, kernel_size=1, bias=score_bias)
        centers = nn.parameter.Parameter(torch.empty([input_dim, K]))
        nn.init.xavier_uniform_(centers)
        self.register_parameter("centers", centers)
        self.intranorm = intranorm
        self.output_dim = input_dim * K

    def forward(self, x):
        b = x.size(0)
        scores = F.softmax(self.score_proj(x), dim=1)
        diff = x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1)
        desc = (scores.unsqueeze(1) * diff).sum(dim=-1)
        if self.intranorm:
            desc = F.normalize(desc, dim=1)
        return F.normalize(desc.view(b, -1), dim=1)


class NetVLADRetriever(Retriever):
    name = "netvlad"

    # Unlike hloc, images are resized to a fixed size so that queries and renders aggregate the
    # same number of local features.
    def __init__(self, device, height=480, width=640, whiten=True, **cfg):
        super().__init__(device, height=height, width=width, whiten=whiten, **cfg)
        import torchvision.models as models
        from scipy.io import loadmat

        mat = loadmat(str(download_checkpoint()), struct_as_record=False, squeeze_me=True)
        layers = mat["net"].layers

        # VGG16 without the classification head and without the last ReLU + MaxPool.
        vgg = list(models.vgg16().children())[0]
        self.backbone = nn.Sequential(*list(vgg.children())[:-2])
        for layer, mat_layer in zip(self.backbone.children(), layers):
            if isinstance(layer, nn.Conv2d):
                # MATLAB stores (S, S, IN, OUT); torch expects (OUT, IN, S, S).
                layer.weight = nn.Parameter(torch.tensor(mat_layer.weights[0]).float().permute([3, 2, 0, 1]))
                layer.bias = nn.Parameter(torch.tensor(mat_layer.weights[1]).float())

        # VLAD layer (index 30); the centers are stored with the opposite sign.
        self.netvlad = NetVLADLayer()
        self.netvlad.score_proj.weight = nn.Parameter(
            torch.tensor(layers[30].weights[0]).float().permute([1, 0]).unsqueeze(-1))
        self.netvlad.centers = nn.Parameter(torch.tensor(-layers[30].weights[1]).float())

        # Whitening layer (index 33) reduces the descriptor to 4096 dimensions.
        self.whiten = None
        if whiten:
            self.whiten = nn.Linear(self.netvlad.output_dim, 4096)
            self.whiten.weight = nn.Parameter(
                torch.tensor(layers[33].weights[0]).float().squeeze().permute([1, 0]))
            self.whiten.bias = nn.Parameter(torch.tensor(layers[33].weights[1].squeeze()).float())
            self.whiten = self.whiten.eval().to(device)

        # NetVLAD works in [0, 255] minus the checkpoint mean image, with unit std.
        self.mean = np.asarray(mat["net"].meta.normalization.averageImage[0, 0], dtype=np.float32)
        self.backbone = self.backbone.eval().to(device)
        self.netvlad = self.netvlad.eval().to(device)
        self.transform = T.Compose([T.Resize((height, width)), T.ToTensor()])

    @torch.inference_mode()
    def embed(self, img_path):
        x = self.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
        x = torch.clamp(x * 255.0, 0.0, 255.0)
        x = x - x.new_tensor(self.mean).view(1, -1, 1, 1)

        feat = self.backbone(x)
        b, c = feat.shape[:2]
        desc = self.netvlad(F.normalize(feat.view(b, c, -1), dim=1))
        if self.whiten is not None:
            desc = F.normalize(self.whiten(desc), dim=1)
        return desc.detach().cpu().numpy().astype("float32")

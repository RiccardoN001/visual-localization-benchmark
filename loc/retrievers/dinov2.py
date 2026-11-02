import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from . import Retriever

HUB_REPO, HUB_MODEL = "facebookresearch/dinov2", "dinov2_vitl14"


class DINOv2Retriever(Retriever):
    name = "dinov2"

    # The global descriptor is the normalized CLS token. Sizes are multiples of the 14-pixel patch.
    def __init__(self, device, height=336, width=448, hub_repo=HUB_REPO, hub_model=HUB_MODEL, **cfg):
        super().__init__(device, height=height, width=width, hub_model=hub_model, **cfg)
        self.model = torch.hub.load(hub_repo, hub_model, trust_repo=True).eval().to(device)
        self.transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def embed(self, img_path):
        x = self.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
        desc = self.model.forward_features(x)["x_norm_clstoken"]
        desc = F.normalize(desc, p=2, dim=-1)
        return desc.detach().cpu().numpy().astype("float32")

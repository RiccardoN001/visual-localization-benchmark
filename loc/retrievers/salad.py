import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from . import Retriever

HUB_REPO, HUB_MODEL = "serizba/salad", "dinov2_salad"


class SaladRetriever(Retriever):
    name = "salad"

    def __init__(self, device, height=322, width=322, hub_repo=HUB_REPO, hub_model=HUB_MODEL, **cfg):
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
        desc = F.normalize(self.model(x), p=2, dim=-1)
        return desc.detach().cpu().numpy().astype("float32")

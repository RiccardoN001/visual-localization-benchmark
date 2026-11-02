import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from . import Retriever

HUB_REPO, HUB_MODEL = "gmberton/MegaLoc", "get_trained_model"


class MegaLocRetriever(Retriever):
    name = "megaloc"

    def __init__(self, device, height=322, width=322, hub_repo=HUB_REPO, hub_model=HUB_MODEL, **cfg):
        super().__init__(device, height=height, width=width, **cfg)
        self.model = torch.hub.load(hub_repo, hub_model, trust_repo=True).eval().to(device)
        # Normalize before resizing, as in the MegaLoc reference code.
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(size=[height, width], antialias=True),
        ])

    @torch.inference_mode()
    def embed(self, img_path):
        x = self.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(self.device)
        desc = F.normalize(self.model(x), p=2, dim=-1)
        return desc.detach().cpu().numpy().astype("float32")

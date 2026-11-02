import torch
import torch.nn.functional as F
from PIL import Image

from . import Retriever

HF_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"   # gated on HuggingFace, needs HF_TOKEN


class DINOv3Retriever(Retriever):
    name = "dinov3"

    # height/width = None keeps the default preprocessing of the checkpoint (224x224).
    def __init__(self, device, height=None, width=None, hf_model=HF_MODEL, **cfg):
        super().__init__(device, height=height, width=width, hf_model=hf_model, **cfg)
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(hf_model)
        self.model = AutoModel.from_pretrained(hf_model).eval().to(device)
        if height and width:
            self.processor.size = {"height": height, "width": width}
            self.processor.crop_size = {"height": height, "width": width}

    @torch.inference_mode()
    def embed(self, img_path):
        x = self.processor(images=Image.open(img_path).convert("RGB"), return_tensors="pt")
        out = self.model(**x.to(self.device))
        # pooler_output is the CLS token when the model provides it.
        desc = getattr(out, "pooler_output", None)
        if desc is None:
            desc = out.last_hidden_state[:, 0]
        desc = F.normalize(desc, p=2, dim=-1)
        return desc.detach().cpu().numpy().astype("float32")

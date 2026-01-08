import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from backbones import get_model

# -------------------------
# Configuration
# -------------------------
IMAGE_SIZE = (112, 112)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Utilities
# -------------------------
def preprocess(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE)

    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor = (img_tensor - 127.5) / 128.0  # ArcFace normalization
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    return img_tensor, img


def load_backbone(model_path):
    backbone = get_model(
        "r100",
        dropout=0.0,
        fp16=True,
        num_features=512
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE, weights_only=False)
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    return backbone


def get_embedding(model, img_tensor):
    with torch.no_grad():
        emb = model(img_tensor)
        emb = torch.nn.functional.normalize(emb)
    return emb.squeeze(0)

# -------------------------
# Main
# -------------------------
def main():
    img1_path = Path(sys.argv[1])

    model = load_backbone("../work_dirs/ms1mv3_r100/model.pt")

    img1_tensor, img1_raw = preprocess(img1_path)
    emb1 = get_embedding(model, img1_tensor)

    #print(f"Embedding (image 1): shape={emb1.shape}")
    print(emb1.cpu().numpy())

if __name__ == "__main__":
    main()


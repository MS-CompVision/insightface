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


def cosine_similarity(e1, e2):
    return torch.dot(e1, e2).item()


def show_single(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_side_by_side(img1, img2, similarity):
    h = max(img1.shape[0], img2.shape[0])

    def pad(img, h):
        if img.shape[0] == h:
            return img
        pad_h = h - img.shape[0]
        return cv2.copyMakeBorder(
            img, 0, pad_h, 0, 0,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    img1 = pad(img1, h)
    img2 = pad(img2, h)

    combined = np.hstack([img1, img2])

    caption = f"Cosine similarity: {similarity:.4f}"

    cv2.imshow(caption, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------
# Main
# -------------------------
def main():
    if len(sys.argv) not in [2,3]:
        print("Usage:")
        print("  python compare_images.py image.jpg")
        print("  python compare_images.py image1.jpg image2.jpg")
        sys.exit(1)

    img1_path = Path(sys.argv[1])
    img2_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None

    model = load_backbone("../work_dirs/ms1mv3_r100/model.pt")

    img1_tensor, img1_raw = preprocess(img1_path)
    emb1 = get_embedding(model, img1_tensor)

    print(f"Embedding (image 1): shape={emb1.shape}")
    print(emb1.cpu().numpy())

    if img2_path is None:
        show_single(img1_raw)
        return

    img2_tensor, img2_raw = preprocess(img2_path)
    emb2 = get_embedding(model, img2_tensor)

    print(f"Embedding (image 2): shape={emb2.shape}")
    print(emb2.cpu().numpy())

    sim = cosine_similarity(emb1, emb2)
    print(f"\nCosine similarity: {sim:.4f}")

    show_side_by_side(img1_raw, img2_raw, sim)


if __name__ == "__main__":
    main()


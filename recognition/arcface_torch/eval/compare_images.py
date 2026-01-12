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
print('Device: ',DEVICE)


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

def euclidean_distance(e1, e2):
    return torch.norm(e1 - e2).item()

def l2_similarity(e1, e2):
    distance = euclidean_distance(e1, e2)
    return 1 - (distance / 2)
    

def resize_keep_aspect(img, target_h=512):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def show_side_by_side(img1, img2, similarity):
    # Resize both images to height 512 while keeping aspect ratio
    img1 = resize_keep_aspect(img1, 512)
    img2 = resize_keep_aspect(img2, 512)

    # If channel mismatch (just in case)
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([img1, img2])

    caption = f"Cosine similarity: {similarity:.4f}"

    cv2.imshow(caption, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------
# Main
# -------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python compare_images.py image1.jpg image2.jpg")
        sys.exit(1)

    img1_path = Path(sys.argv[1])
    img2_path = Path(sys.argv[2])

    model = load_backbone("../work_dirs/ms1mv3_r100/model.pt")

    img1_tensor, img1_raw = preprocess(img1_path)
    emb1 = get_embedding(model, img1_tensor)

    #print(f"Embedding (image 1): shape={emb1.shape}")
    # print(emb1.cpu().numpy())

    img2_tensor, img2_raw = preprocess(img2_path)
    emb2 = get_embedding(model, img2_tensor)

    #print(f"Embedding (image 2): shape={emb2.shape}")
    # print(emb2.cpu().numpy())

    sim1 = cosine_similarity(emb1, emb2)
    sim2 = euclidean_distance(emb1, emb2)

    show_side_by_side(img1_raw, img2_raw, sim1)
    print(f"\nEuclidian similarity: {sim2:.4f}")


if __name__ == "__main__":
    main()


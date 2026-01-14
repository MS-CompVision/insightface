import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

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
    

def resize_keep_aspect(img, target_h):
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def show_side_by_side(img1, img2, similarity):
    # Resize to height 512 with aspect ratio preserved
    img1 = resize_keep_aspect(img1, 512)
    img2 = resize_keep_aspect(img2, 512)

    # Convert grayscale → RGB for matplotlib
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    combined = np.hstack([img1, img2])

    plt.figure(figsize=(12, 6))
    plt.imshow(combined)
    plt.axis("off")
    plt.title(f"Cosine similarity: {similarity:.4f}", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)

# Visualize difference between two embeddings using PyTorch (CUDA compatible).
def visualize_embedding_difference(enc1, enc2, shape=(32, 16), device=None):
    # Device handling
    if device is None:
        device = enc1.device
    else:
        device = torch.device(device)

    enc1 = enc1.to(device).flatten()
    enc2 = enc2.to(device).flatten()

    if enc1.numel() != shape[0] * shape[1]:
        raise ValueError(
            f"Embedding size {enc1.numel()} does not match shape {shape}"
        )

    # Reshape
    m1 = enc1.view(*shape)
    m2 = enc2.view(*shape)

    # Absolute difference
    diff = torch.abs(m1 - m2)

    # Normalize diff for colormap
    diff_norm = diff / (diff.max() + 1e-8)

    # Move to CPU for plotting
    m1 = m1.cpu()
    m2 = m2.cpu()
    diff_norm = diff_norm.cpu()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---- Embedding 1 ----
    axes[0].imshow(m1, cmap="gray")
    axes[0].set_title("Embedding 1")
    for i in range(shape[0]):
        for j in range(shape[1]):
            axes[0].text(j, i, f"{m1[i, j]:.2f}",
                          ha="center", va="center", fontsize=6)

    # ---- Embedding 2 ----
    axes[1].imshow(m2, cmap="gray")
    axes[1].set_title("Embedding 2")
    for i in range(shape[0]):
        for j in range(shape[1]):
            axes[1].text(j, i, f"{m2[i, j]:.2f}",
                          ha="center", va="center", fontsize=6)

    # ---- Difference heatmap ----
    # Blue (close to 0) -> Red (far from 0)
    im = axes[2].imshow(diff_norm, cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    axes[2].set_title("|Embedding₁ − Embedding₂|")

    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

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

    visualize_embedding_difference(emb1,emb2,(32, 16),DEVICE)


if __name__ == "__main__":
    main()


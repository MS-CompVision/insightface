import argparse
import math
import os

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def get_augmentations(img_size):
    """
    Returns:
      - individual augmentations for visualization
      - full InsightFace training pipeline (as provided)
    """

    individual = {
        "Original": transforms.Compose([
            transforms.Resize((img_size, img_size)),
        ]),

        "RandomResizedCrop": transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        ]),

        "HorizontalFlip": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
        ]),

        "Rotation_10deg": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=10),
        ]),

        "ColorJitter": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.03),
        ]),

        "GaussianBlur": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1.5)),
        ]),

        "RandomErasing": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=1.0, scale=(0.03, 0.2), ratio=(0.3, 3.0)),
            transforms.ToPILImage(),
        ]),
    }

    full_pipeline = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.03),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.3, scale=(0.03, 0.2), ratio=(0.3, 3.0)),
        transforms.ToPILImage(),
    ])

    return individual, full_pipeline


def make_grid(images, titles, save_path=None):
    n = len(images)
    cols = 3
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 4, rows * 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[✓] Saved grid to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to input face image (positional)")
    parser.add_argument("--img_size", type=int, default=112, help="Target image size (InsightFace default is 112)")
    parser.add_argument("--out", type=str, default="augmentation_grid.png", help="Output grid image")
    args = parser.parse_args()

    assert os.path.exists(args.image), f"Image not found: {args.image}"

    img = Image.open(args.image).convert("RGB")

    individual, full_pipeline = get_augmentations(args.img_size)

    images = []
    titles = []

    for name, tfm in individual.items():
        out_img = tfm(img)
        images.append(out_img)
        titles.append(name)

    full_img = full_pipeline(img)
    images.append(full_img)
    titles.append("Full_Pipeline")

    make_grid(images, titles, save_path=args.out)


if __name__ == "__main__":
    main()
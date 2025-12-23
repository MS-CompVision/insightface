import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split('\t')
        pairs.append(parts)
    return pairs


def read_paths(root_dir, pairs):
    paths = []
    labels = []
    for pair in pairs:
        if len(pair) == 3:  # same person
            name = pair[0]
            img1 = os.path.join(root_dir, name, f"{name}_{int(pair[1]):04d}.jpg")
            img2 = os.path.join(root_dir, name, f"{name}_{int(pair[2]):04d}.jpg")
            issame = True
        else:  # different people
            name1 = pair[0]
            name2 = pair[2]
            img1 = os.path.join(root_dir, name1, f"{name1}_{int(pair[1]):04d}.jpg")
            img2 = os.path.join(root_dir, name2, f"{name2}_{int(pair[3]):04d}.jpg")
            issame = False

        paths.append((img1, img2))
        labels.append(issame)
    return paths, labels


def load_image(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img)


def get_embedding(model, img):
    model.eval()
    with torch.no_grad():
        emb = model(img.unsqueeze(0).cuda())
        emb = F.normalize(emb)
    return emb.cpu().numpy().flatten()


def evaluate(model, data_dir, pairs_file):
    """
    Generic evaluation function for LFW/CFP-FP/AgeDB-30.
    """
    pairs = read_pairs(pairs_file)
    paths, labels = read_paths(data_dir, pairs)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    embeddings1 = []
    embeddings2 = []

    for p1, p2 in paths:
        img1 = load_image(p1, transform)
        img2 = load_image(p2, transform)

        emb1 = get_embedding(model, img1)
        emb2 = get_embedding(model, img2)

        embeddings1.append(emb1)
        embeddings2.append(emb2)

    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    labels = np.array(labels)

    sims = np.sum(embeddings1 * embeddings2, axis=1)

    # Simple threshold search
    best_acc = 0
    for t in np.linspace(-1, 1, 1000):
        pred = sims > t
        acc = (pred == labels).mean()
        best_acc = max(best_acc, acc)

    return best_acc


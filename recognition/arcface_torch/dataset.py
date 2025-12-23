import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, root_dir, img_size=112):
        self.root_dir = root_dir
        self.samples = []
        self.class_to_idx = {}
        self.img_size = img_size

        folders = sorted(os.listdir(root_dir))
        for idx, folder in enumerate(folders):
            full = os.path.join(root_dir, folder)
            if not os.path.isdir(full):
                continue
            self.class_to_idx[folder] = idx
            for fname in os.listdir(full):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(full, fname), idx))

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        print(f"[dataset] Found {len(self.samples)} images across {len(self.class_to_idx)} identities in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def get_dataloader(
    rec,            # dataset root (cfg.rec)
    local_rank,     # int
    batch_size,     # int
    dali,           # bool (ignored here)
    dali_aug,       # bool (ignored here)
    seed,           # int
    num_workers     # int
):
    """
    Signature intentionally matches train_v2.py call:
      get_dataloader(cfg.rec, local_rank, cfg.batch_size, cfg.dali, cfg.dali_aug, cfg.seed, cfg.num_workers)

    This implementation ignores DALI flags and uses standard PyTorch DataLoader.
    """

    # deterministic-ish behavior
    torch.manual_seed(int(seed))

    dataset = FaceDataset(rec, img_size=112)

    # distributed sampler if torch.distributed initialized
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=local_rank,
            shuffle=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )

    return loader


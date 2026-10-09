import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split, StratifiedKFold

CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]
LABEL2ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}

def get_transforms(img_size=224, norm_mean=(0.5, 0.5, 0.5), norm_std=(0.5, 0.5, 0.5), use_augmentation=False):
    """
    Menyiapkan transformasi citra untuk training dan evaluasi.
    
    Catatan Saintifik LC25000:
    Dataset LC25000 (Borkowski et al., 2019) aslinya berasal dari 750 citra biopsi patologi
    yang telah diaugmentasi secara offline menjadi 15.000 citra (5.000 per kelas) menggunakan
    rotasi dan flip.
    
    Oleh karena itu, secara default use_augmentation=False agar model melatih citra murni
    apa adanya (Resize + ToTensor + Normalize) tanpa distorsi rotasi/color-jitter berlebihan.
    Jika use_augmentation=True, augmentasi on-the-fly (flip, rotasi 20, affine, jitter) diaktifkan.
    """
    if use_augmentation:
        train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.RandomAffine(degrees=0, scale=(0.9, 1.1), shear=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])
    else:
        train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std)
        ])

    eval_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

    return train_transform, eval_transform

class LungHistDataset(Dataset):
    def __init__(self, df, transform=None):
        self.filepaths = df["filepath"].values
        self.labels = df["label_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def load_or_create_splits(dataset_dir, output_dir=None, seed=42, test_ratio=0.2, splits_file=None):
    if splits_file and os.path.exists(splits_file):
        df_all = pd.read_csv(splits_file)
        train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
        test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)
        print(f"[OK] Memuat master splits dari {splits_file}: Train={len(train_df)}, Test={len(test_df)}")
        if output_dir and not os.path.exists(os.path.join(output_dir, "splits.csv")):
            os.makedirs(output_dir, exist_ok=True)
            df_all.to_csv(os.path.join(output_dir, "splits.csv"), index=False)
        return train_df, test_df

    if output_dir and os.path.exists(os.path.join(output_dir, "splits.csv")):
        df_all = pd.read_csv(os.path.join(output_dir, "splits.csv"))
        train_df = df_all[df_all["split"] == "train"].reset_index(drop=True)
        test_df = df_all[df_all["split"] == "test"].reset_index(drop=True)
        print(f"[OK] Memuat splits.csv yang ada: Train={len(train_df)}, Test={len(test_df)}")
        return train_df, test_df

    filepaths = []
    labels = []
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")

    for root, dirs, files in os.walk(dataset_dir):
        for cls_name in CLASS_NAMES:
            if os.path.basename(root) == cls_name:
                for ext in valid_exts:
                    for f in sorted(glob.glob(os.path.join(root, ext))):
                        filepaths.append(os.path.abspath(f))
                        labels.append(cls_name)

    if len(filepaths) == 0:
        raise FileNotFoundError(f"Tidak ada citra ditemukan di {dataset_dir}. Pastikan folder kelas berisi lung_aca, lung_n, lung_scc.")

    df_all = pd.DataFrame({"filepath": filepaths, "label": labels})
    df_all["label_id"] = df_all["label"].map(LABEL2ID)

    train_df, test_df = train_test_split(
        df_all,
        test_size=test_ratio,
        stratify=df_all["label_id"],
        random_state=seed,
        shuffle=True
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df["split"] = "train"
    test_df["split"] = "test"
    combined_df = pd.concat([train_df, test_df], ignore_index=True)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        combined_df.to_csv(os.path.join(output_dir, "splits.csv"), index=False)
        print(f"[OK] File splits.csv berhasil dibuat dan disimpan di {output_dir}")

    print(f"[OK] Dataset LC25000 berhasil dimuat: Total={len(df_all)} citra | Train={len(train_df)} | Test={len(test_df)}")
    return train_df, test_df

from pathlib import Path
from PIL import Image
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

class SISRDataset(Dataset):
    def __init__(self, root, crop=128, scale=2, train=True):
        self.paths = sorted([
            *Path(root).glob("*.png"),
            *Path(root).glob("*.jpg"),
            *Path(root).glob("*.jpeg"),
        ])
        self.crop = crop
        self.scale = scale
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.train:
            x = random.randint(0, img.width - self.crop)
            y = random.randint(0, img.height - self.crop)
            hr = img.crop((x, y, x + self.crop, y + self.crop))
        else:
            hr = TF.center_crop(img, (self.crop, self.crop))

        lr = hr.resize((self.crop // self.scale,) * 2, Image.BICUBIC)
        return {
            "input": TF.to_tensor(lr),
            "target": TF.to_tensor(hr)
        }

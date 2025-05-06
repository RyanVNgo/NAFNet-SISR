
import os

from .custom_dataset import SISRDataset
from torch.utils.data import Dataset, DataLoader


def setup_dataloaders(options, batch_size, device):
    patch_size = options.get('patch_size', 128)
    scale = options.get('scale', 2)
    workers = options.get('workers', 4)
    train_set_opts = options.get('training')
    valid_set_opts = options.get('validation')
    train_set_path = os.path.expanduser(train_set_opts.get('data_path'))
    valid_set_path = os.path.expanduser(valid_set_opts.get('data_path'))
    train_dataset = SISRDataset(train_set_path, patch_size, scale)
    valid_dataset = SISRDataset(valid_set_path, patch_size, scale)

    train_loader = DataLoader(
        train_dataset, 
        batch_size, 
        shuffle=True, 
        num_workers=workers,
        persistent_workers=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=True
    )
    return {'train':train_loader , 'valid':valid_loader}



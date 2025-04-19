from .custom_dataset import SISRDataset
from torch.utils.data import Dataset, DataLoader

def get_loader(data_path, batch_size=16, train=True):
    dataset = SISRDataset(data_path, train=train)
    return DataLoader(dataset, batch_size, shuffle=True)


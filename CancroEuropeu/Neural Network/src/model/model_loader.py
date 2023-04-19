from torch.utils.data import DataLoader
from dataclasses import dataclass

@dataclass
class ModelLoader:
    loader: any
    images: any
    labels: any

    def __init__(self, loader:DataLoader, images:DataLoader, labels:DataLoader) -> None:
        self.loader = loader
        self.images = images
        self.labels = labels
        pass
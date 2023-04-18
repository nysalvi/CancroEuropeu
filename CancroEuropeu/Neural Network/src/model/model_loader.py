from dataclasses import dataclass

@dataclass
class ModelLoader:
    loader: any
    images: any
    labels: any

    def __init__(self, loader, images, labels) -> None:
        self.loader = loader
        self.images = images
        self.labels = labels
        pass
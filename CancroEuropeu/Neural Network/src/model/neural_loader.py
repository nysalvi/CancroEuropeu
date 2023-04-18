from dataclasses import dataclass
from src.model.model_loader import ModelLoader

@dataclass
class NeuralLoader:
    train_loader: ModelLoader
    dev_loader : ModelLoader
    test_loader: ModelLoader

    def __init__(self, train_loader, train_images, train_labels, dev_loader, dev_images, dev_labels, 
            test_loader, test_images, test_labels) -> None:
        self.train_loader = ModelLoader(train_loader, train_images, train_labels)
        self.dev_loader = ModelLoader(dev_loader, dev_images, dev_labels)
        self.test_loader = ModelLoader(test_loader, test_images, test_labels)
        pass
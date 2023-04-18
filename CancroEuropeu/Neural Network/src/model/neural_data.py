from dataclasses import dataclass

@dataclass
class NeuralData:
    train_data: any
    test_data: any

    def __init__(self, train_images, test_images) -> None:
        self.train_data = train_images
        self.test_data = test_images
        pass
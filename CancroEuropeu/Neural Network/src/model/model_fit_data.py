from dataclasses import dataclass

@dataclass
class ModelFitData:
    optimizer_ft: any
    model_ft: any

    def __init__(self, optimizer_ft, model_ft) -> None:
        self.optimizer_ft = optimizer_ft
        self.model_ft = model_ft
        pass
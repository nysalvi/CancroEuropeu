from array import array
from src.model.enums.model_name_enum import ModelName

class Models:
    @staticmethod
    def select_models(model_name: ModelName) -> array:
        models = []
        if model_name == ModelName.ALL:
            for model in ModelName:
                if model != ModelName.ALL:
                    models.append(model)
        else:
            models.append(model_name)
        return models
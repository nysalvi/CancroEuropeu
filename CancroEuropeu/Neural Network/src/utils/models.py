from array import array
from ..model.enums.model_name_enum import ModelName

class Models:
    @staticmethod
    def select_models(model_name: ModelName|str) -> array:
        models = []
        if type(model_name) == ModelName:
            if model_name == ModelName.ALL:
                for model in ModelName:
                    if model != ModelName.ALL:
                        models.append(model)
            else:
                models.append(model_name)
            return models
        elif type(model_name) == str:
            if model_name == ModelName.ALL.name:
                for model in ModelName:
                    if model != ModelName.ALL.name:
                        models.append(ModelName[model_name])
            else:
                models.append(ModelName[model_name])
            return models


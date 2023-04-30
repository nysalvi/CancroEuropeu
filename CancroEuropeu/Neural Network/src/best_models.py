from .utils.global_info import Info
from .model.enums.model_name_enum import ModelName
from .services.analysis import Analysis
from .services.evaluation import Evaluation

if __name__ == "__main__":
    Analysis.best_thresolds()
    Evaluation.best_results()

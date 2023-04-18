import gdown
import dotenv
import os

class ModelInitialize:

    @staticmethod
    def initialize():
        file_path = "models/model.pth"
        if not os.path.exists(file_path):
            dotenv.load_dotenv(dotenv.find_dotenv())
            url = os.getenv("model_url")
            gdown.download(url, file_path, quiet=False)

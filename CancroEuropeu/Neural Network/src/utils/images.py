import gdown
import zipfile
import dotenv
import os

class Images:
    @staticmethod
    def download(outputDirectory, nameFile):
        dotenv.load_dotenv(dotenv.find_dotenv())
        url = os.getenv("images_url")
        output = outputDirectory + nameFile
        gdown.download(url, output, quiet=False)

    @staticmethod
    def unzip(directory, nameFile):
        output = directory + nameFile
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(directory)
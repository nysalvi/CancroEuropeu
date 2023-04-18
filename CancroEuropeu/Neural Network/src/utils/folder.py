from genericpath import exists
import os

class Folder:
    @staticmethod
    def createFolder(path, name):
        if path is None:
            path = os.getcwd()
        directory = name
        parent_dir = path
        fullPath = os.path.join(parent_dir, directory)
        os.makedirs(fullPath, exist_ok = True)

    @staticmethod
    def getCurrentFolder():
        return os.getcwd()
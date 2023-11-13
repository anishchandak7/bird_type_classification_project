# Import necessary libraries.
import os
from pathlib import Path
import logging

# set logging basic configuration.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


PROJECT_NAME = 'cnnClassifier'

# Project folder structure.
project_files_list = [
    ".github/workflow/.gitkeep",
    f"src/{PROJECT_NAME}/components/__init__.py",
    f"src/{PROJECT_NAME}/config/__init__.py",
    f"src/{PROJECT_NAME}/config/configuration.py",
    f"src/{PROJECT_NAME}/constants/__init__.py",
    f"src/{PROJECT_NAME}/entity/__init__.py",
    f"src/{PROJECT_NAME}/pipeline/__init__.py",
    f"src/{PROJECT_NAME}/utils/__init__.py",
    f"src/{PROJECT_NAME}/__init__.py",
    "config/config.yaml",
    "templates/index.html",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb"
]

def generate_folder_structure():

    """
    This function is responsible for generating the project folder structure based on the list of files provided in `project_files_list`.
    """

    for file_path in project_files_list:

        # Convert 'file_path' to Path Object
        path = Path(file_path)
        directory, file_name = os.path.split(path)

        # Create directory if not exists.
        if directory != "":
            os.makedirs(directory, exist_ok=True)
            logging.info("New %s created successfully!", directory)

        # Create an empty file if not exists.
        if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            with open(path, 'w', encoding='utf-8') as f:
                pass
            logging.info("%s created successfully!", file_name)
        else:
            logging.warning("%s already exists.", file_name)

# Calling generate folder structure function.
generate_folder_structure()

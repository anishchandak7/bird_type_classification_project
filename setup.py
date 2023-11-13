# Import necessary libraries.
import setuptools

# Reading README file for long description.
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Package version
__version__ = "0.0.0"

# Other details.
REPO_NAME = "bird_type_classification_project"
AUTHOR_USER_NAME = "anishchandak7"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "chandakanish7@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
import os, sys, requests
from zipfile import ZipFile
from io import BytesIO

if not os.path.exists("prostate_cancer_prediction/data"):
    os.mkdir("prostate_cancer_prediction/data")

temp = requests.get("https://zenodo.org/records/10775529/files/_database.zip?download=1")
temp = ZipFile(BytesIO(temp.content))
temp.extractall("prostate_cancer_prediction/data")
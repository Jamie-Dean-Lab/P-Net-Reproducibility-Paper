import os, sys, requests
from zipfile import ZipFile
from io import BytesIO

if not os.path.exists("Prostate Cancer Prediction/data"):
    os.mkdir("Prostate Cancer Prediction/data")

temp = requests.get("https://zenodo.org/records/10775529/files/_database.zip?download=1")
temp = ZipFile(BytesIO(temp.content))
temp.extractall("Prostate Cancer Prediction/data")
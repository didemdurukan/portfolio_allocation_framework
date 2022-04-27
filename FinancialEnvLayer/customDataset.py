# User Imports his/her own dataset
from dataset import Dataset
import pandas as pd
import json


class ImportCustomDataset(Dataset):

    def __init__(self, path):

        self.df = pd.DataFrame()
        self.path = path
        self.createDataset()

    def createDataset(self):

        csv = ".csv"
        excel = ".xlsx"
        json_str = ".json"

        try:
            if csv in self.path:
                data = pd.read_csv(self.path)
            elif excel in self.path:
                data = pd.read_excel(self.path)
            elif json_str in self.path:
                f = open(self.path)
                json_data = json.load(f)
                data = pd.json_normalize(json_data)

            self.df = data
            return self.df

        except FileNotFoundError:
            print("File not found.")

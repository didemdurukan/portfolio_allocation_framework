# Test File
from customDataset import ImportCustomDataset
import pandas as pd

if __name__ == '__main__':

    df = ImportCustomDataset("examplejson.json").createDataset()
    print(df)

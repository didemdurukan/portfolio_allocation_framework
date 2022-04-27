#ABC Example

from abc import ABC, abstractmethod
import pandas as pd
 
class Dataset(ABC):

    df = pd.DataFrame()
    extended_df = pd.DataFrame()

    def __init__(self,df,extended_df):
        self.df = df
        self.extended_df = extended_df

    @abstractmethod
    def createDataset(self):
        pass

    
       
import pandas as pd
from torch.utils.data import Dataset
from sklearn import preprocessing
import torch
import ast

from typing import Union


class SemevalDataset(Dataset):

    def __init__(self, csv_file: str) -> None:
        roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        label_encoder = preprocessing.LabelEncoder()
        self.dataframe = pd.read_csv(csv_file)

        #self.dataframe['vector'] = label_encoder.fit_transform(self.dataframe.vector.values)
        self.tokens = [roberta.encode(x, y) for x, y in zip(self.dataframe['chunk1'], self.dataframe['chunk2'])]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: Union[int, slice, str]) -> tuple:
        x_data = self.tokens
        vector = self.dataframe.iloc[idx, 2]
        vector = ast.literal_eval(vector)
        #vector2 = [n.strip() for n in vector]
        #lista = list(vector)
        value_data = torch.Tensor(vector)
        #value_data = torch.Tensor(self.dataframe.iloc[:, 2].values)
        # exp_data = torch.Tensor(self.dataframe.iloc[:, 3].values)

        return x_data[idx], value_data#, exp_data[idx]
    
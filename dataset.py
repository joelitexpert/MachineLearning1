from sklearn.datasets import make_moons
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

class CreditCardDataset(Dataset):
    def __init__(self, path="./creditcard_2023.csv", set="train", random_state=0):
        super().__init__()  

        df = pd.read_csv(path)
        self.x = np.array(df.drop(columns=["Class", "id"]))
        self.y = np.array(df["Class"])
        

        N = self.x.shape[0]
        

        np.random.seed(random_state)
        perm = np.random.permutation(N)

        self.x = self.x[perm]
        self.y = self.y[perm]

        print(self.y.mean())

        num_train = int(N * 0.7)
        num_val = int(N * 0.1)

        if set == "train":
            self.x = self.x[:num_train]
            self.y = self.y[:num_train]
        elif set == "val":
            self.x = self.x[num_train:num_train + num_val]
            self.y = self.y[num_train:num_train + num_val]
        else:
            self.x = self.x[num_train + num_val:]
            self.y = self.y[num_train + num_val:]

    def get_mean(self):
        return np.mean(self.x, axis=0).astype(np.float32)[None, ...]

    def get_std(self):
        return np.std(self.x, axis=0).astype(np.float32)[None, ...]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index].astype(np.float32), self.y[index]

if __name__ == "__main__":
    dataset = CreditCardDataset()
